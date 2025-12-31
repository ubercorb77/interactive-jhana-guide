# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a jhana meditation assistant with two interaction modes:
1. **Smile biofeedback** - Detects smiles via webcam, sends scores to Claude for guidance
2. **Voice interaction** - Speech-based conversation with Claude, similar to ChatGPT voice mode

## Architecture

The project has four main components:

1. **Smile Detection** (`smile-detector/`) - MediaPipe face mesh to detect mouth landmarks and calculate a 0-100 smile score based on mouth width and corner lift, normalized to face height. Uses calibration (neutral face + biggest smile samples) for per-user accuracy.

2. **Smile-Based Meditation** (`jhana-asmr-loop/`) - Smile biofeedback application:
   - Runs smile detection in the OpenCV main loop
   - Tracks smile scores over a time window (deque with timestamps)
   - Background thread sends average smile scores to Claude Haiku every N seconds
   - Claude responses are converted to audio via Replicate's Kokoro TTS and played through pygame

3. **Voice-Based Meditation** (`guided-meditation/`) - Voice interaction application:
   - Records speech via `sounddevice` with energy-based voice activity detection
   - Transcribes via OpenAI Whisper API
   - Sends to Claude Haiku for meditation guidance
   - On silence (10s), sends `[SILENCE]` to Claude; Claude responds or returns `[WAIT]` to give more space
   - TTS via Replicate Kokoro, playback via pygame

4. **Audio Generation** (`kokoro-asmr-generation/`) - Replicate API testing for Kokoro TTS model (`jaaari/kokoro-82m`, voice `af_nicole`)

## Key Dependencies

- `opencv-python`, `mediapipe` - Face detection and landmark tracking (smile mode)
- `sounddevice`, `scipy` - Audio recording (voice mode)
- `openai` - Whisper API for speech-to-text (voice mode)
- `anthropic` - Claude API client (uses `claude-haiku-4-5-20251001`)
- `replicate` - TTS audio generation (Kokoro)
- `pygame` - Audio playback
- `python-dotenv` - Environment variable loading

## Environment Setup

API keys are loaded from `../../.env` (two directories up):
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY` (for voice mode)
- Replicate API key (via `replicate` library default)

## Running

### Voice Mode (recommended)
```bash
cd guided-meditation
python claude_jhana_voice_v1.py
```
Speak anytime, or stay silent. Claude will check in every 10s of silence. Press `Ctrl+C` to end.

### Smile Mode
```bash
cd jhana-asmr-loop
python claude_jhana_asmr_loop_v1.py
```
Controls: `P` to calibrate (neutral then smile), `S` to start session, `Q` to quit.

## Configuration

### Voice Mode (`claude_jhana_voice_v1.py`)
- `SILENCE_CHECK_INTERVAL` - Seconds before Claude decides to speak or wait (default: 10)
- `VAD_THRESHOLD` - Energy threshold for speech detection (default: 0.02)

### Smile Mode (`claude_jhana_asmr_loop_v1.py`)
- `LOOP_DELAY` - Seconds between feedback cycles (default: 20)
- `SMILE_WINDOW` - Seconds for averaging smile scores (default: 5)
- `TOTAL_DURATION` - Session length in seconds (default: 20 minutes)

## File Naming Convention

Python files are prefixed with `claude_` and versioned (e.g., `claude_smile_detect_v2.py`).
