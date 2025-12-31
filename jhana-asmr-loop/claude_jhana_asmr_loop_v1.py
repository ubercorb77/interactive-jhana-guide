import warnings
import os
import logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import time
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
import replicate
import pygame
from termcolor import colored

# ====== CONFIGURATION ======
LOOP_DELAY = 20  # seconds to wait after audio finishes before next feedback
SMILE_WINDOW = 5  # X/2 seconds - window for calculating average smile
TOTAL_DURATION = 20 * 60  # Y minutes converted to seconds (20 minutes default)
INITIAL_PROMPT_FILE = "initial_prompt_v5.txt"


# ====== SETUP ======

log_file = None

def log(message):
    """append message to log file"""
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

screen_width, screen_height = pyautogui.size()
# print(screen_width, screen_height)

# initialize pygame for audio playback
pygame.mixer.init()

# smile score tracking
smile_scores = deque()  # (timestamp, score) tuples
smile_lock = threading.Lock()

# meditation session state
session_start_time = None
session_active = False

# initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# mouth landmark indices
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
UPPER_LIP = 0
LOWER_LIP = 17
FOREHEAD_TOP = 10
CHIN_BOTTOM = 152

# calibration states
WAIT_FOR_START = 0
CALIBRATE_NEUTRAL = 1
WAIT_FOR_SMILE = 2
CALIBRATE_SMILE = 3
CALIBRATED = 4

# calibration data
calibration_state = WAIT_FOR_START
calibration_data = {
    'neutral_width': [],
    'neutral_lift': [],
    'smile_width': [],
    'smile_lift': [],
    'min_width': None,
    'max_width': None,
    'min_lift': None,
    'max_lift': None
}

def calculate_distance(point1, point2):
    """calculate euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_raw_metrics(landmarks, image_width, image_height):
    """
    calculate raw mouth metrics without scoring
    returns: mouth_width_to_face_height, corner_lift_to_face_height
    """
    # get key mouth points
    mouth_left = landmarks[MOUTH_LEFT]
    mouth_right = landmarks[MOUTH_RIGHT]
    mouth_top = landmarks[MOUTH_TOP]
    mouth_bottom = landmarks[MOUTH_BOTTOM]
    
    # get forehead and chin for normalization
    forehead = landmarks[FOREHEAD_TOP]
    chin = landmarks[CHIN_BOTTOM]
    forehead_y = forehead.y * image_height
    chin_y = chin.y * image_height
    face_height = abs(chin_y - forehead_y)
    
    # convert normalized coordinates to pixel coordinates
    left_x = mouth_left.x * image_width
    left_y = mouth_left.y * image_height
    right_x = mouth_right.x * image_width
    right_y = mouth_right.y * image_height
    top_y = mouth_top.y * image_height
    bottom_y = mouth_bottom.y * image_height

    # calculate mouth width
    mouth_width = calculate_distance([left_x, left_y], [right_x, right_y])
    
    # calculate corner lift
    center_y = (top_y + bottom_y) / 2
    avg_corner_y = (left_y + right_y) / 2
    corner_lift = center_y - avg_corner_y
    
    # normalize to face height
    mouth_width_to_face_height = mouth_width / face_height
    corner_lift_to_face_height = corner_lift / face_height
    
    return mouth_width_to_face_height, corner_lift_to_face_height

def linear_map(value, min_val, max_val):
    """map value from [min_val, max_val] to [0, 100]"""
    if max_val == min_val:
        return 50  # avoid division by zero
    return (value - min_val) / (max_val - min_val) * 100

def calculate_smile_score(landmarks, image_width, image_height, debug=False):
    """
    calculate smile score from 0-100 based on calibration
    """
    width_metric, lift_metric = calculate_raw_metrics(landmarks, image_width, image_height)
    
    # if not calibrated, return raw value
    if calibration_state != CALIBRATED:
        return 0, {'width_metric': width_metric, 'lift_metric': lift_metric}
    
    # map to 0-100 using calibration
    width_score = linear_map(width_metric, 
                             calibration_data['min_width'], 
                             calibration_data['max_width'])
    lift_score = linear_map(lift_metric, 
                            calibration_data['min_lift'], 
                            calibration_data['max_lift'])
    
    # average the scores
    smile_score = (width_score + lift_score) / 2
    
    if debug:
        return int(smile_score), {
            'width_metric': width_metric,
            'lift_metric': lift_metric,
            'width_score': width_score,
            'lift_score': lift_score,
            'min_width': calibration_data['min_width'],
            'max_width': calibration_data['max_width'],
            'min_lift': calibration_data['min_lift'],
            'max_lift': calibration_data['max_lift']
        }
    
    return int(smile_score), {}

def finalize_calibration():
    """calculate min/max from calibration samples"""
    if len(calibration_data['neutral_width']) > 0 and len(calibration_data['smile_width']) > 0:
        all_widths = calibration_data['neutral_width'] + calibration_data['smile_width']
        all_lifts = calibration_data['neutral_lift'] + calibration_data['smile_lift']
        
        calibration_data['min_width'] = min(all_widths)
        calibration_data['max_width'] = max(all_widths)
        calibration_data['min_lift'] = min(all_lifts)
        calibration_data['max_lift'] = max(all_lifts)
        
        print(f"\ncalibration complete! ^^")
        print(f"width range: {calibration_data['min_width']:.2f} to {calibration_data['max_width']:.2f}")
        print(f"lift range: {calibration_data['min_lift']:.4f} to {calibration_data['max_lift']:.4f}\n")
        return True
    return False

def get_average_smile_score():
    """calculate average smile score from the past SMILE_WINDOW seconds"""
    with smile_lock:
        current_time = time.time()
        cutoff_time = current_time - SMILE_WINDOW
        
        # filter scores within the time window
        recent_scores = [score for timestamp, score in smile_scores if timestamp >= cutoff_time]
        
        if recent_scores:
            return sum(recent_scores) / len(recent_scores)
        return 0

def generate_and_play_audio(text):
    """generate audio from text using replicate and play it"""
    try:
        print(colored(f"generating audio...", 'magenta'), end=' ', flush=True)
        
        input_data = {
            "text": text,
            "voice": "af_nicole"
        }
        
        output = replicate.run(
            "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13",
            input=input_data
        )
        
        # save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"meditation_{timestamp}.wav"
        
        with open(f"audios/{output_filename}", "wb") as file:
            file.write(output.read())
        
        # play audio
        print(colored(f"playing {output_filename}...", 'magenta'), end=' ', flush=True)
        log(f"playing audio: {output_filename}")
        pygame.mixer.music.load(f"audios/{output_filename}")
        pygame.mixer.music.play()
        
        print(colored("audio started ^^", 'magenta'))
        return output_filename  # return filename for cleanup
        
    except Exception as e:
        print(f"error with audio: {e} >_<")
        return None


def meditation_feedback_loop():
    """main loop that sends smile scores to claude and plays responses"""
    global session_active
    
    while session_active:
        current_time = time.time()
        elapsed = current_time - session_start_time
        
        # check if session is over
        if elapsed >= TOTAL_DURATION:
            print(f"\nmeditation session complete! {TOTAL_DURATION/60:.0f} minutes ^^")
            session_active = False
            break

        # wait before next cycle
        print(f"waiting {LOOP_DELAY} seconds before next feedback...")
        time.sleep(LOOP_DELAY)
        
        # get average smile score
        avg_score = get_average_smile_score()
        
        print(colored(f"\n--- feedback cycle (elapsed: {elapsed/60:.2f}min) ---", 'green'))
        log(f"--- feedback cycle (elapsed: {elapsed/60:.2f}min) ---")
        print(colored(f"avg smile score in recent {SMILE_WINDOW} seconds: {avg_score:.1f}", 'yellow'))
        log(f"avg smile score in recent {SMILE_WINDOW} seconds: {avg_score:.1f}")
        
        # send to claude
        user_message = f"smile score: {avg_score:.1f}"
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            message = client.messages.create(
                max_tokens=1024,
                messages=conversation_history,
                model="claude-haiku-4-5-20251001",
            )
            
            claude_response = message.content[0].text
            conversation_history.append({
                "role": "assistant",
                "content": claude_response
            })
            
            print(colored(f"claude: {claude_response}", 'blue'))
            log(f"claude: {claude_response}")

            # generate and play audio
            audio_file = generate_and_play_audio(claude_response)
            
            # wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # # clean up audio file
            # if audio_file and os.path.exists(audio_file):
            #     os.remove(audio_file)
            
        except Exception as e:
            print(f"error with claude api: {e} >_<")
            time.sleep(5)  # wait a bit before retrying


# start webcam
cap = cv2.VideoCapture(0)

# create named window
window_name = 'jhana meditation guide ^^'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# the code for centering window doesn't really work but whatever, it's okay
window_width = 1280
window_height = 720

# calculate center position
x_pos = (screen_width - window_width) // 2
y_pos = (screen_height - window_height) // 2

# move window to center
cv2.moveWindow(window_name, x_pos, y_pos)

# debug mode toggle
DEBUG_MODE = True

print("=== jhana meditation guide with smile detection ===")
print("step 1: press 'P' to start capturing NEUTRAL face (50 samples)")
print("step 2: press 'P' again to capture BIGGEST SMILE (50 samples)")
print("step 3: press 'S' to start meditation session")
print(f"step 4: meditate for {TOTAL_DURATION/60:.0f} minutes! press 'q' to quit early")
print()

# initialize log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/meditation_log_{timestamp}.txt"
os.makedirs("logs", exist_ok=True)  # create logs folder if it doesn't exist

# initialize anthropic client
load_dotenv("../../.env")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# conversation history for claude
conversation_history = []

# load initial prompt
with open(INITIAL_PROMPT_FILE, 'r') as f:
    initial_prompt = f.read()
log("initial prompt:")
log(initial_prompt)

conversation_history.append({
    "role": "user",
    "content": initial_prompt
})

# get initial response from claude
print("getting initial response from claude...")
log("getting initial response from claude...")
message = client.messages.create(
    max_tokens=1024,
    messages=conversation_history,
    model="claude-haiku-4-5-20251001",
)
initial_response = message.content[0].text
conversation_history.append({
    "role": "assistant",
    "content": initial_response
})
print(colored(f"claude: {initial_response}\n", 'blue'))
log(f"claude: {initial_response}")

feedback_thread = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("failed to grab frame >_<")
        break
    
    # flip frame horizontally
    frame = cv2.flip(frame, 1)
    
    # convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # process the frame
    results = face_mesh.process(rgb_frame)
    
    # get frame dimensions
    h, w, _ = frame.shape
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # get raw metrics
            width_metric, lift_metric = calculate_raw_metrics(face_landmarks.landmark, w, h)
            
            # store calibration data
            if calibration_state == CALIBRATE_NEUTRAL:
                calibration_data['neutral_width'].append(width_metric)
                calibration_data['neutral_lift'].append(lift_metric)
                if len(calibration_data['neutral_width']) >= 50:
                    print(f"neutral face captured! (50 samples)")
                    print("now press 'P' to capture your BIGGEST SMILE ^^")
                    calibration_state = WAIT_FOR_SMILE
            elif calibration_state == CALIBRATE_SMILE:
                calibration_data['smile_width'].append(width_metric)
                calibration_data['smile_lift'].append(lift_metric)
                if len(calibration_data['smile_width']) >= 50:
                    print(f"smile captured! (50 samples)")
                    if finalize_calibration():
                        calibration_state = CALIBRATED
                        print("now press 'S' to start meditation session!")
            
            # calculate smile score
            smile_score, debug_info = calculate_smile_score(face_landmarks.landmark, w, h, debug=DEBUG_MODE)
            
            # store smile score if session is active
            if session_active:
                with smile_lock:
                    smile_scores.append((time.time(), smile_score))
                    # keep only recent scores
                    cutoff_time = time.time() - SMILE_WINDOW * 2
                    while smile_scores and smile_scores[0][0] < cutoff_time:
                        smile_scores.popleft()
            
            # draw face mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
            
            # highlight mouth landmarks
            mouth_landmarks = [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM]
            for landmark_idx in mouth_landmarks:
                landmark = face_landmarks.landmark[landmark_idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
            
            # display calibration instructions or smile score
            if calibration_state == WAIT_FOR_START:
                cv2.putText(frame, "press 'P' to start capturing NEUTRAL face", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            elif calibration_state == CALIBRATE_NEUTRAL:
                cv2.putText(frame, "calibration: hold NEUTRAL face...", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"samples: {len(calibration_data['neutral_width'])}/50", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif calibration_state == WAIT_FOR_SMILE:
                cv2.putText(frame, "press 'P' again to capture BIGGEST SMILE", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            elif calibration_state == CALIBRATE_SMILE:
                cv2.putText(frame, "calibration: hold BIGGEST SMILE!", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"samples: {len(calibration_data['smile_width'])}/50", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:  # CALIBRATED
                if session_active:
                    elapsed = time.time() - session_start_time
                    remaining = TOTAL_DURATION - elapsed
                    cv2.putText(frame, f"meditation in progress... {remaining/60:.1f}min left", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "press 'S' to start meditation session", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                # change color based on score
                if smile_score < 30:
                    color = (128, 128, 128)
                    text = "neutral"
                elif smile_score < 60:
                    color = (0, 255, 255)
                    text = "slight smile"
                else:
                    color = (0, 255, 0)
                    text = "big smile!"
                
                cv2.putText(frame, f"smile score: {smile_score}", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(frame, text, 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # draw debug info
            if DEBUG_MODE:
                y_offset = 200
                line_height = 30
                cv2.putText(frame, f"width_metric: {debug_info.get('width_metric', 0):.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"lift_metric: {debug_info.get('lift_metric', 0):.4f}", 
                           (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if calibration_state == CALIBRATED:
                    cv2.putText(frame, f"width_score: {debug_info.get('width_score', 0):.1f}", 
                               (10, y_offset + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"lift_score: {debug_info.get('lift_score', 0):.1f}", 
                               (10, y_offset + line_height*3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "no face detected >_<", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # show the frame
    cv2.imshow(window_name, frame)
    
    # handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nshutting down...")
        session_active = False
        pygame.mixer.music.stop()
        time.sleep(0.5)  # give thread time to exit cleanly
        break
    elif key == ord('p') or key == ord('P'):
        if calibration_state == WAIT_FOR_START:
            print("capturing neutral face... hold still!")
            calibration_state = CALIBRATE_NEUTRAL
        elif calibration_state == WAIT_FOR_SMILE:
            print("capturing smile... hold that smile!")
            calibration_state = CALIBRATE_SMILE
    elif key == ord('s') or key == ord('S'):
        if calibration_state == CALIBRATED and not session_active:
            print(f"\nstarting meditation session for {TOTAL_DURATION/60:.0f} minutes! ^^")
            
            # play initial response and wait for it to finish
            audio_file = generate_and_play_audio(initial_response)
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            # NOW set session active after audio finishes
            session_active = True
            session_start_time = time.time()
            
            # # clean up audio file
            # if audio_file and os.path.exists(audio_file):
            #     os.remove(audio_file)
            
            # now start feedback thread
            feedback_thread = threading.Thread(target=meditation_feedback_loop, daemon=True)
            feedback_thread.start()

# cleanup
session_active = False
if feedback_thread:
    feedback_thread.join(timeout=2)

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
pygame.mixer.quit()

print("\nsession ended. thank you for meditating! ^^")