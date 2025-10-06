from transformers import pipeline
import soundfile as sf

# Загружаем аудиофайл
audio_file = "audio.wav"
data, samplerate = sf.read(audio_file)

# Загружаем модель
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="jonatasgrosman/wav2vec2-large-xlsr-53-russian"
)

# Распознаём речь
result = asr_pipeline({"array": data, "sampling_rate": samplerate})
print(f"Распознанный текст: {result['text']}")
