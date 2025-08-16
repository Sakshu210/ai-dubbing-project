# In ai_pipeline.py

import os
import subprocess
import torch
import whisper
import soundfile as sf
import noisereduce as nr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

# --- 1. Load All Models Once (CPU Version) ---

print("--- Setting device to CPU ---")
device = "cpu"

print("Loading all AI models. This may take a few minutes...")

# Whisper Model
print("Loading Whisper model...")
whisper_model = whisper.load_model("medium", device=device)

# NLLB Translation Model
print("Loading NLLB translation model...")
translator_model_name = "facebook/nllb-200-distilled-600M"
translator_tokenizer = AutoTokenizer.from_pretrained(translator_model_name, src_lang="hin_Deva")
translator_model = AutoModelForSeq2SeqLM.from_pretrained(translator_model_name)

# Coqui TTS Model
print("Loading Coqui TTS model...")
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print("✅ All models loaded successfully!")


# --- 2. Define the Main Pipeline Function ---

def process_video_pipeline(video_path: str) -> str:
    """
    This function takes the path to an input video and processes it through the entire pipeline.
    Returns the path to the final dubbed video.
    """
    print("--- Starting AI Dubbing Pipeline ---")

    # Define paths for all intermediate files
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    extracted_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
    cleaned_audio_path = os.path.join(temp_dir, "cleaned_audio.wav")
    hindi_transcript_path = os.path.join(temp_dir, "hindi_transcript.txt")
    english_transcript_path = os.path.join(temp_dir, "english_translation.txt")
    dubbed_audio_path = os.path.join(temp_dir, "dubbed_english.wav")
    final_video_path = os.path.join(temp_dir, "final_dubbed_video.mp4")

    # Step A: Extract Audio from Video using FFmpeg
    print("Step 1/6: Extracting audio from video...")
    # The 'subprocess.run' command is the standard way to run external commands in Python
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", "-y", extracted_audio_path
    ], check=True)

    # Step B: Noise Reduction
    print("Step 2/6: Reducing audio noise...")
    data, rate = sf.read(extracted_audio_path)
    reduced_noise_data = nr.reduce_noise(y=data, sr=rate)
    sf.write(cleaned_audio_path, reduced_noise_data, rate)

    # Step C: Transcription
    print("Step 3/6: Transcribing audio to Hindi text...")
    result = whisper_model.transcribe(cleaned_audio_path)
    hindi_text = result["text"]
    with open(hindi_transcript_path, "w", encoding="utf-8") as f:
        f.write(hindi_text)
    print(f" > Hindi Text: {hindi_text}")

    # Step D: Translation
    print("Step 4/6: Translating text to English...")
    inputs = translator_tokenizer(hindi_text, return_tensors="pt")
    translated_tokens = translator_model.generate(
        **inputs,
        forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=100
    )
    english_text = translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    with open(english_transcript_path, "w", encoding="utf-8") as f:
        f.write(english_text)
    print(f" > English Text: {english_text}")

    # Step E: Text-to-Speech (Voice Cloning)
    print("Step 5/6: Synthesizing English audio with cloned voice...")
    tts_model.tts_to_file(
        text=english_text,
       speaker_wav=cleaned_audio_path,
        language="en",
        file_path=dubbed_audio_path
    )
    
    # Step F: Lip Synchronization
    print("Step 6/6: Performing lip synchronization...")
    # Note: Wav2Lip requires its own environment. We assume it's set up.
    # The paths here must be relative to where the script is run or absolute.
    wav2lip_command = [
        "python", "Wav2Lip/inference.py",
        "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth",
        "--face", video_path,
        "--audio", dubbed_audio_path,
        "--outfile", final_video_path
    ]
    subprocess.run(wav2lip_command, check=True)

    print("--- Pipeline Finished Successfully! ---")
    return final_video_path