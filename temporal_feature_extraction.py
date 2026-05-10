import librosa
import numpy as np

def extract_temporal_features(
    file_path,
    duration=20, 
    n_mfcc=13,
    n_fft=2048,
    hop_length=512
):
    """
        Extract temporal features from an audio file

        Input Parameters:
        file_path (str): Full path to the audio file 
        duration (sec):  Duration of audio content to load (in seconds)
        n_mfcc (int >0):    Number of MFCC coefficients
        n_fft (int >0):     FFT size = Samples per time-frame
        hop_length (int > 0): Number of samples to slide the FFT window

        Output Returned:
        T_duration              -- Total duration of audio content (in seconds)
        T_frame : float         -- Duration of each time-frame (in seconds)
        n_frames : int > 0      -- Number of (overlapping) time-frames (in seconds)
        features : np.1darray   --  feature vector of shape (n_frames,)
        
        

    """
    # Load audio
    y, sr = librosa.load(file_path, duration=duration)
    features = librosa.feature.spectral_centroid(y=y, sr=sr)
    T_audio = len(y)/sr                                             # Duraction of audio content (in seconds)    
    T_frame = n_fft / sr                                            # Duration of each time-frame (in seconds)
    n_frames = floor(1 + (len(y) - n_fft) / hop_length)             # Number of (overlapping) time-frames
                                                
    return T_audio, T_frame, n_frames, features 


#   TEST FEATURE EXTRACTION
if __name__ == "__main__":
    test_audio = "audio/yaman.wav"
    T_audio, T_frame, n_frames, features = extract_temporal_features(test_audio)
    print(f"Actual duration of the processed audio content: {T_audio} seconds")
    print(f"Duration of each time-frame: {T_frame} seconds")
    print(f"Number of time-frames processed: {n_frames}")
    print("Temporal Feature vector shape:", features.shape)
    print("First 5 spectral-centroid values:", features[:5])
