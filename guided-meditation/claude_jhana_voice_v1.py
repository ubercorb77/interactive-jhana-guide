import warnings
import os
import logging
import io
import time
import threading
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
import replicate
import pygame
from termcolor import colored

# ====== CONFIGURATION ======
SILENCE_CHECK_INTERVAL = 20  # seconds before asking claude if it wants to speak
SAMPLE_RATE = 16000  # whisper expects 16kHz
SILENCE_DURATION = 3.0  # seconds of silence to consider speech ended
MIN_SPEECH_DURATION = 0.5  # minimum seconds of speech to process
PROMPT_FILE = "voice_prompt.txt"

# calibration settings
CALIBRATION_DURATION = 1.5  # seconds to sample ambient noise
VAD_THRESHOLD_MULTIPLIER = 3.0  # threshold = ambient_avg * this multiplier

# debug settings
DEBUG_VOLUME = True  # show live volume meter
VOLUME_BAR_WIDTH = 40  # width of volume bar in characters

# these get set during calibration
VAD_THRESHOLD = 0.02  # will be overwritten by calibration
MAX_DISPLAY_ENERGY = 0.15  # will be overwritten by calibration

# ====== SETUP ======
load_dotenv("../../.env")

def get_builtin_mic():
    """find the built-in microphone device index"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        name = device['name'].lower()
        # look for macbook built-in mic
        if 'macbook' in name and device['max_input_channels'] > 0:
            return i, device['name']
        if 'built-in' in name and 'microphone' in name and device['max_input_channels'] > 0:
            return i, device['name']
    # fallback: look for any built-in input
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if 'built-in' in name and device['max_input_channels'] > 0:
            return i, device['name']
    return None, None

# find and set the built-in mic
BUILTIN_MIC_INDEX, BUILTIN_MIC_NAME = get_builtin_mic()
if BUILTIN_MIC_INDEX is not None:
    sd.default.device[0] = BUILTIN_MIC_INDEX  # set default input device
    print(f"using mic: {BUILTIN_MIC_NAME} (device {BUILTIN_MIC_INDEX})")
else:
    print("warning: could not find built-in mic, using system default")

# initialize clients
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# initialize pygame for audio playback
pygame.mixer.init()

# create directories
os.makedirs("audios", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# logging
log_file = None

def init_log():
    """initialize log file for this session"""
    global log_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/session_{timestamp}.md"

    with open(log_file, 'w') as f:
        f.write(f"# session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    return log_file

def write_log_config(vad_threshold, silence_interval):
    """write config section to log"""
    if not log_file:
        return

    with open(log_file, 'a') as f:
        f.write("## config\n\n")
        f.write(f"- vad_threshold: {vad_threshold:.4f}\n")
        f.write(f"- silence_check_interval: {silence_interval}s\n\n")
        f.write("---\n\n")
        f.write("## conversation\n\n")

def write_log(role, content):
    """append message to log file - exact API format"""
    if not log_file:
        return

    timestamp = datetime.now().strftime("%H:%M:%S")

    with open(log_file, 'a') as f:
        f.write(f"**[{timestamp}] {role}:** {content}\n\n")

# session state
session_active = False
is_playing_audio = False
last_user_speech_time = time.time()
audio_lock = threading.Lock()

# conversation history
conversation_history = []


def log(message):
    """print with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def calculate_energy(audio_chunk):
    """calculate RMS energy of audio chunk"""
    return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))


def calibrate_vad():
    """calibrate VAD threshold based on ambient noise"""
    global VAD_THRESHOLD, MAX_DISPLAY_ENERGY

    print(f"\ncalibrating... please stay quiet for {CALIBRATION_DURATION}s")

    energy_samples = []

    def calibration_callback(indata, frames, time_info, status):
        audio_chunk = indata[:, 0]
        energy = calculate_energy(audio_chunk)
        energy_samples.append(energy)

    # record ambient noise
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=int(SAMPLE_RATE * 0.1), callback=calibration_callback):
        start_time = time.time()
        while time.time() - start_time < CALIBRATION_DURATION:
            # show progress
            elapsed = time.time() - start_time
            progress = int((elapsed / CALIBRATION_DURATION) * 20)
            bar = "█" * progress + "░" * (20 - progress)
            print(f"\r[{bar}] {elapsed:.1f}s / {CALIBRATION_DURATION}s", end="", flush=True)
            time.sleep(0.05)

    print()  # newline after progress bar

    if not energy_samples:
        print("calibration failed, using defaults")
        return

    # calculate stats
    avg_energy = np.mean(energy_samples)
    max_energy = np.max(energy_samples)

    # set threshold above ambient noise
    VAD_THRESHOLD = avg_energy * VAD_THRESHOLD_MULTIPLIER

    # set display max to show headroom above threshold
    MAX_DISPLAY_ENERGY = VAD_THRESHOLD * 5

    print(f"calibration complete:")
    print(f"  - ambient avg: {avg_energy:.4f}")
    print(f"  - ambient max: {max_energy:.4f}")
    print(f"  - VAD threshold: {VAD_THRESHOLD:.4f} ({VAD_THRESHOLD_MULTIPLIER}x ambient)")
    print()


def print_volume_bar(energy, threshold, is_speaking):
    """print a live volume meter"""
    # scale energy for display
    normalized = min(energy / MAX_DISPLAY_ENERGY, 1.0)
    bar_length = int(normalized * VOLUME_BAR_WIDTH)
    threshold_pos = int((threshold / MAX_DISPLAY_ENERGY) * VOLUME_BAR_WIDTH)

    # build the bar
    bar = ""
    for i in range(VOLUME_BAR_WIDTH):
        if i < bar_length:
            if i >= threshold_pos:
                bar += colored("█", "green")  # above threshold
            else:
                bar += colored("█", "yellow")  # below threshold
        elif i == threshold_pos:
            bar += colored("|", "red")  # threshold marker
        else:
            bar += "░"

    status = colored("SPEAKING", "green") if is_speaking else colored("silent", "grey")
    print(f"\r[{bar}] {energy:.4f} {status}   ", end="", flush=True)


def record_speech():
    """record audio until silence is detected, return numpy array"""
    log("listening...")

    audio_buffer = []
    silence_start = None
    speech_detected = False
    speech_start = None
    current_energy = 0.0

    def audio_callback(indata, frames, time_info, status):
        nonlocal silence_start, speech_detected, speech_start, current_energy

        audio_chunk = indata[:, 0]  # mono
        energy = calculate_energy(audio_chunk)
        current_energy = energy

        if energy > VAD_THRESHOLD:
            if not speech_detected:
                speech_detected = True
                speech_start = time.time()
                if DEBUG_VOLUME:
                    print()  # newline before "speech detected"
                log("speech detected...")
            silence_start = None
            audio_buffer.append(audio_chunk.copy())
        elif speech_detected:
            audio_buffer.append(audio_chunk.copy())
            if silence_start is None:
                silence_start = time.time()

    # start recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=int(SAMPLE_RATE * 0.1), callback=audio_callback):
        while True:
            # show volume meter
            if DEBUG_VOLUME:
                print_volume_bar(current_energy, VAD_THRESHOLD, speech_detected)

            time.sleep(0.05)

            # check if we should stop
            if speech_detected and silence_start:
                if time.time() - silence_start > SILENCE_DURATION:
                    if DEBUG_VOLUME:
                        print()  # newline after volume bar
                    break

            # timeout if no speech for a long time
            if not speech_detected and len(audio_buffer) == 0:
                if time.time() - last_user_speech_time > SILENCE_CHECK_INTERVAL:
                    if DEBUG_VOLUME:
                        print()  # newline after volume bar
                    return None  # signal silence timeout

    if not audio_buffer:
        return None

    # check minimum duration
    audio_data = np.concatenate(audio_buffer)
    duration = len(audio_data) / SAMPLE_RATE

    if duration < MIN_SPEECH_DURATION:
        log(f"speech too short ({duration:.1f}s), ignoring")
        return None

    log(f"recorded {duration:.1f}s of audio")
    return audio_data


def transcribe_audio(audio_data):
    """transcribe audio using OpenAI Whisper API"""
    try:
        # convert to int16 for wav
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # write to bytes buffer
        buffer = io.BytesIO()
        wavfile.write(buffer, SAMPLE_RATE, audio_int16)
        buffer.seek(0)
        buffer.name = "audio.wav"

        # transcribe
        log("transcribing...")
        response = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,
            language="en"
        )

        text = response.text.strip()
        log(colored(f"you: {text}", 'green'))
        write_log("user", text)
        return text

    except Exception as e:
        log(f"transcription error: {e}")
        return None


def get_claude_response(user_message):
    """get response from claude"""
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    try:
        message = anthropic_client.messages.create(
            max_tokens=256,
            messages=conversation_history,
            model="claude-haiku-4-5-20251001",
        )

        response = message.content[0].text.strip()
        conversation_history.append({
            "role": "assistant",
            "content": response
        })

        write_log("assistant", response)

        return response

    except Exception as e:
        log(f"claude error: {e}")
        return None


def generate_and_play_audio(text):
    """generate audio from text using replicate and play it"""
    global is_playing_audio

    try:
        log(colored("generating audio...", 'magenta'))

        output = replicate.run(
            "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13",
            input={"text": text, "voice": "af_nicole"}
        )

        # save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"audios/voice_{timestamp}.wav"

        with open(output_filename, "wb") as file:
            file.write(output.read())

        # play audio
        log(colored("playing...", 'magenta'))
        with audio_lock:
            is_playing_audio = True

        pygame.mixer.music.load(output_filename)
        pygame.mixer.music.play()

        # wait for audio to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        with audio_lock:
            is_playing_audio = False

        log("audio finished")

    except Exception as e:
        log(f"audio error: {e}")
        with audio_lock:
            is_playing_audio = False


def handle_silence():
    """called when silence timeout reached - ask claude if it wants to speak"""
    global last_user_speech_time

    write_log("user", "[SILENCE]")
    response = get_claude_response("[SILENCE]")

    if response and response.strip() != "[WAIT]":
        log(colored(f"claude: {response}", 'blue'))
        generate_and_play_audio(response)
    else:
        log("claude chose to wait...")

    last_user_speech_time = time.time()


def main_loop():
    """main conversation loop"""
    global session_active, last_user_speech_time

    # load initial prompt
    with open(PROMPT_FILE, 'r') as f:
        initial_prompt = f.read()

    # write config to log
    write_log_config(VAD_THRESHOLD, SILENCE_CHECK_INTERVAL)

    conversation_history.append({
        "role": "user",
        "content": initial_prompt
    })

    # log the initial prompt as user message
    write_log("user", initial_prompt)

    # get initial greeting
    log("getting initial response from claude...")
    initial_response = get_claude_response("")  # empty to trigger assistant response
    # remove the empty user message we just added
    conversation_history.pop()
    conversation_history.pop()

    # properly get first response
    message = anthropic_client.messages.create(
        max_tokens=256,
        messages=conversation_history,
        model="claude-haiku-4-5-20251001",
    )
    initial_response = message.content[0].text.strip()
    conversation_history.append({
        "role": "assistant",
        "content": initial_response
    })

    log(colored(f"claude: {initial_response}", 'blue'))
    generate_and_play_audio(initial_response)

    last_user_speech_time = time.time()
    session_active = True

    log("\n" + "="*50)
    log("meditation session started! speak anytime, or stay silent.")
    log("press Ctrl+C to end session")
    log("="*50 + "\n")

    while session_active:
        try:
            # check if audio is playing
            with audio_lock:
                if is_playing_audio:
                    time.sleep(0.1)
                    continue

            # check for silence timeout
            time_since_speech = time.time() - last_user_speech_time
            if time_since_speech >= SILENCE_CHECK_INTERVAL:
                handle_silence()
                continue

            # try to record speech (with timeout)
            audio_data = record_speech()

            if audio_data is None:
                # silence timeout occurred during recording
                handle_silence()
                continue

            # transcribe
            text = transcribe_audio(audio_data)
            if not text:
                continue

            last_user_speech_time = time.time()

            # get claude response
            response = get_claude_response(text)
            if response:
                log(colored(f"claude: {response}", 'blue'))
                generate_and_play_audio(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"error in main loop: {e}")
            time.sleep(1)


def main():
    global session_active

    print("\n" + "="*50)
    print("  jhana voice meditation guide")
    print("="*50)
    print(f"\nconfig:")
    print(f"  - silence check interval: {SILENCE_CHECK_INTERVAL}s")
    print(f"  - sample rate: {SAMPLE_RATE}Hz")
    print(f"  - debug volume: {DEBUG_VOLUME}")

    # initialize logging
    log_path = init_log()
    print(f"  - log file: {log_path}")

    # calibrate VAD threshold based on ambient noise
    calibrate_vad()

    if DEBUG_VOLUME:
        print("volume bar legend:")
        print(f"  {colored('█', 'yellow')} = below threshold (not detected as speech)")
        print(f"  {colored('█', 'green')} = above threshold (detected as speech)")
        print(f"  {colored('|', 'red')} = VAD threshold marker")
        print()

    try:
        main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        session_active = False
        pygame.mixer.quit()
        print("\n\nsession ended. thank you for meditating! ^^")
        print(f"log saved to: {log_file}")


if __name__ == "__main__":
    main()
