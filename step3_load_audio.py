import librosa

audio_path = "audio/Nilambari.wav"

y, sr = librosa.load(audio_path, duration=30)

print("Sample rate:", sr)
print("Audio shape:", y.shape)
print("First 10 samples:", y[:10])

print("Duration (seconds):", len(y) / sr)

