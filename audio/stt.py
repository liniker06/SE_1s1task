import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

audio, sr = librosa.load("Recording.wav", sr=16000)
processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small").to("cpu")

inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
generated_ids = model.generate(inputs)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(text)
