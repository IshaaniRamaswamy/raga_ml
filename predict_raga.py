import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from feature_extraction import extract_features
from dataset_loader import load_dataset
from sklearn.model_selection import train_test_split

# -------------------------
# Load dataset & train model
# -------------------------
X, y = load_dataset("dataset")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/test split (for demonstration, we use all data for training)
X_train = X
y_train = y_encoded

model = SVC(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)

# -------------------------
# Load new audio file
# -------------------------
# Replace with your path, e.g. "audio/yaman.wav"
if len(sys.argv) > 1:
    new_audio_file = sys.argv[1]
else:
    new_audio_file = "audio/yaman.wav"

features = extract_features(new_audio_file).reshape(1, -1)

# -------------------------
# Predict raga
# -------------------------
pred_encoded = model.predict(features)
pred_raga = encoder.inverse_transform(pred_encoded)

print(f"Predicted Raga: {pred_raga[0]}")

