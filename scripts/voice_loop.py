#!/usr/bin/env python3
"""
Minimal always-on voice loop client for JARVISv3 (Headless Mode).
- Continuously records short audio windows from the system microphone
- Sends each window to /api/v1/voice/session (STT->Chat->TTS)
- If meaningful text is detected, plays assistant audio
- Supports simple barge-in: stops playback if user keeps talking (conceptually, though this script is simple loop)

Requirements (host environment):
  pip install sounddevice numpy simpleaudio requests

Usage:
  python scripts/voice_loop.py --host http://localhost:8000 --window-sec 1.5 --rate 16000
"""
import argparse
import base64
import io
import json
import sys
import threading
import time
import wave
from typing import Optional

import requests

try:
    import sounddevice as sd
    import numpy as np
except ImportError as e:
    print("[voice_loop] Missing dependency: sounddevice/numpy (pip install sounddevice numpy). Error:", e)
    sys.exit(1)

try:
    import simpleaudio as sa
except ImportError as e:
    print("[voice_loop] Missing dependency: simpleaudio (pip install simpleaudio). Error:", e)
    sys.exit(1)


def float_to_wav_bytes(audio: np.ndarray, samplerate: int) -> bytes:
    """Convert float32 mono audio (-1..1) to 16-bit PCM WAV bytes."""
    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # clip and scale
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000", help="Backend host URL")
    ap.add_argument("--window-sec", type=float, default=2.0, help="Recording window in seconds")
    ap.add_argument("--rate", type=int, default=16000, help="Sample rate (Hz)")
    ap.add_argument("--device", default=None, help="Input device name or index (default system)")
    ap.add_argument("--mode", default="chat", help="Chat mode: chat|coding|research")
    ap.add_argument("--include-web", action="store_true", help="Include web search")
    args = ap.parse_args()

    session_url = f"{args.host.rstrip('/')}/api/v1/voice/session"

    # Track conversation across turns
    conversation_id: Optional[str] = None

    # Playback controller
    play_lock = threading.Lock()
    current_play: Optional[sa.PlayObject] = None

    def stop_playback():
        nonlocal current_play
        with play_lock:
            try:
                if current_play is not None and current_play.is_playing():
                    current_play.stop()
            except Exception:
                pass
            current_play = None

    def play_wav_bytes(wav_bytes: bytes):
        nonlocal current_play
        try:
            with play_lock:
                # Stop any existing playback (barge-in)
                if current_play is not None and current_play.is_playing():
                    current_play.stop()
                wave_obj = sa.WaveObject.from_wave_read(wave.open(io.BytesIO(wav_bytes)))
                current_play = wave_obj.play()
                # Wait for playback to finish before listening again?
                # For true barge-in, we shouldn't wait, but we need VAD to know when to interrupt.
                # Here we wait to avoid re-recording the assistant's voice.
                current_play.wait_done() 
        except Exception as e:
            print("[voice_loop] Playback error:", e)

    # Configure input stream
    samplerate = args.rate
    channels = 1
    blocksize = int(args.window_sec * samplerate)

    print(f"[voice_loop] Connecting to {session_url}")
    print("[voice_loop] Starting. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Record a short window
            # print(".", end="", flush=True)
            try:
                audio = sd.rec(frames=blocksize, samplerate=samplerate, channels=channels, dtype="float32", device=args.device)
                sd.wait()
            except Exception as e:
                print("\n[voice_loop] Recording error:", e)
                time.sleep(1)
                continue

            # Basic VAD: check volume
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01: # Silence threshold
                continue

            print("\n[voice_loop] Processing audio...")
            wav_bytes = float_to_wav_bytes(audio.flatten(), samplerate)
            
            payload = {
                "audio_data": base64.b64encode(wav_bytes).decode("utf-8"),
                "conversation_id": conversation_id,
                "mode": args.mode,
                "include_web": args.include_web
            }
            
            try:
                r = requests.post(session_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=60)
                if r.status_code != 200:
                    print(f"[voice_loop] API Error {r.status_code}: {r.text[:100]}")
                    time.sleep(0.5)
                    continue
                
                resp = r.json()
            except Exception as e:
                print(f"[voice_loop] Request failed: {e}")
                time.sleep(0.5)
                continue

            if not resp.get("detected"):
                # No speech/wake word detected
                continue

            # Update conversation state
            conversation_id = resp.get("conversation_id") or conversation_id
            text_resp = resp.get("text_response", "")
            print(f"[voice_loop] Assistant: {text_resp}")

            # Play audio if available
            audio_b64 = resp.get("audio_data")
            if audio_b64:
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    play_wav_bytes(audio_bytes)
                except Exception as e:
                    print("[voice_loop] Could not play audio:", e)

    except KeyboardInterrupt:
        print("\n[voice_loop] Stopping...")
        stop_playback()
        return

if __name__ == "__main__":
    main()
