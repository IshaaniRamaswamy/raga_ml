import os
import numpy as np
from feature_extraction import extract_features

def load_dataset(dataset_path):
    """
    Load dataset and extract features.

    Returns:
    X (np.ndarray): feature matrix
    y (np.ndarray): raga labels
    """
    X = []
    y = []

    for raga_name in os.listdir(dataset_path):
        raga_path = os.path.join(dataset_path, raga_name)

        if not os.path.isdir(raga_path):
            continue

        for file in os.listdir(raga_path):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(raga_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(raga_name)

    return np.array(X), np.array(y)


# 🔍 TEST DATASET LOADER
#if __name__ == "__main__":
   # X, y = load_dataset("dataset")
   # print("Total samples:", len(X))
   # print("Feature shape:", X.shape)
   # print("First label:", y[0])

if __name__ == "__main__":
    X, y = load_dataset("dataset")
    print("Total samples loaded:", len(X))
    print("Labels:", y)
    
#print(np.unique(y))

