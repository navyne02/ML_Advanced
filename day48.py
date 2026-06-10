import torch
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf

print("--- Step 1: Speech-to-Text (Whisper AI - Listening) ---")
# Loading Whisper for transcription
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

print("--- Step 2: Text-to-Speech (SpeechT5 AI - Speaking) ---")
# Loading SpeechT5 components
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifi_gan")

# Load a speaker embedding (to sound like a human)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Generate Speech
text = "Vanakkaam Naveen! Advanced AI engineering is fascinating."
inputs = processor(text=text, return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# Save the generated audio file
sf.write("speech.wav", speech.numpy(), samplerate=16000)
print(f"✅ Generated audio: 'speech.wav' for text: '{text}'")

print("\n--- Step 3: Verifying with Whisper (AI Listening to itself) ---")
transcription = whisper("speech.wav")["text"]
print(f"👂 AI listened to itself and heard: '{transcription}'")

if text.lower().strip(".") in transcription.lower():
    print("\n🌟 SUCCESS: AI perfectly understood its own generated speech!")