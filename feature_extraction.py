import librosa
import numpy as np

def extract_features(file_path, duration=30, n_mfcc=20):
    """
    Extract MFCC features from an audio file.

    Parameters:
    file_path (str): path to audio file
    duration (int): seconds to load
    n_mfcc (int): number of MFCC coefficients

    Returns:
    np.ndarray: feature vector of shape (n_mfcc,)
    """
    y, sr = librosa.load(file_path, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = np.mean(mfcc.T, axis=0)
    return features


# 🔍 TEST FEATURE EXTRACTION
if __name__ == "__main__":
    test_audio = "audio/yaman.wav"
    features = extract_features(test_audio)
    print("MFCC feature vector shape:", features.shape)
    print("First 5 MFCC values:", features[:5])

