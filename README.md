# Jhana Meditation Assistant

A jhana meditation assistant with two interaction modes: smile biofeedback and voice interaction.

## Folders

### `jhana-asmr-loop/`
Smile biofeedback meditation application. Runs smile detection via OpenCV. Every 20 seconds, tracks average smile score over the last 5 seconds, and sends to Claude Haiku. Claude's guidance is converted to audio via Kokoro TTS and played through pygame.

### `guided-meditation/`
Voice-based meditation interaction. Records speech with voice activity detection, transcribes locally via whisper.cpp, and sends to Claude Haiku for guidance. Every 10 seconds of silence, sends `[SILENCE]` to Claude, who either responds or returns `[WAIT]` to give more space. Uses Kokoro TTS for audio output.

### `smile-detector/`
MediaPipe-based smile detection using face mesh landmarks. Calculates a 0-100 smile score based on mouth width and corner lift, normalized to face height. Includes calibration for per-user accuracy (neutral face + biggest smile samples).

### `kokoro-audio-generation/`
Replicate API testing for Kokoro TTS model (`jaaari/kokoro-82m`, voice `af_nicole`). Contains sample outputs and API experimentation notebook.

## Quick Start

### Voice Mode (recommended)
```bash
cd guided-meditation
python claude_jhana_voice_v1.py
```

### Smile Mode
```bash
cd jhana-asmr-loop
python claude_jhana_asmr_loop_v1.py
```
Controls: `P` to calibrate, `S` to start session, `Q` to quit.

## Environment

API keys loaded from `../../.env`:
- `ANTHROPIC_API_KEY`
- Replicate API key (via library default)
