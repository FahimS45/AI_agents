# main.py

import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

import asyncio
import queue
import sounddevice as sd
import numpy as np
import torch

from stt_tts_loader import stt_model, tts_model, device
from Voice_agent.multiagent import conversation_agent

# --- Audio Queue & Callback ---
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

mic_stream = sd.InputStream(callback=audio_callback, samplerate=16000, channels=1)
mic_stream.start()

# --- Voice Interaction Loop ---
async def voice_loop(agent_workflow):
    buffer = []
    pause_threshold = 0.8  # seconds of silence
    last_voice_time = asyncio.get_event_loop().time()

    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)

        # Check for pause (end of speech)
        if np.abs(chunk).mean() < 0.001:  # near silence
            if asyncio.get_event_loop().time() - last_voice_time > pause_threshold:
                if buffer:
                    audio_data = np.concatenate(buffer, axis=0).flatten()
                    buffer = []

                    # --- STT ---
                    print("Transcribing...")
                    # Whisper expects float32 PCM
                    audio_tensor = torch.from_numpy(audio_data).float()
                    transcription = stt_model.transcribe(audio_tensor.numpy(), fp16=False)["text"]
                    print("User said:", transcription)

                    # --- Agent workflow ---
                    async for response_text in agent_workflow.run(transcription):
                        print("Agent says:", response_text)

                        # --- TTS ---
                        tts_model.tts_to_file(
                            text=response_text,
                            file_path="response.wav"
                        )

                        # Play the response
                        sd.play(sd.read("response.wav")[0], samplerate=22050)
                        sd.wait()
            else:
                await asyncio.sleep(0.01)
        else:
            last_voice_time = asyncio.get_event_loop().time()
            await asyncio.sleep(0.01)

# --- Main ---
async def main():
    class VoiceAgent:
        async def run(self, text):
            yield f"You said: {text}"

    await voice_loop(conversation_agent)

if __name__ == "__main__":
    asyncio.run(main())
