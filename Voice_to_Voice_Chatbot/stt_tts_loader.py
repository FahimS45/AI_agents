# stt_tts_loader.py

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from TTS.api import TTS

# --- Device configuration ---
device = "cuda" #if torch.cuda.is_available() else "cpu"

# --- Load Whisper Base from Hugging Face ---
print("Loading Whisper Base (Hugging Face)...")
whisper_model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(whisper_model_name)
stt_model = WhisperForConditionalGeneration.from_pretrained(
    whisper_model_name
).to(device).eval()  # float32 by default
print("Whisper Base loaded.")

# --- Load Coqui TTS ---
print("Loading Coqui TTS...")
tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC_ph").to(device)
print("Coqui TTS loaded.")