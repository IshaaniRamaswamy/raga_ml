import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from dataset_loader import load_dataset


# Load dataset
X, y = load_dataset("dataset")

print("Original labels:", np.unique(y))


# Encode labels (string → integer)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

#print("Encoded labels:", np.unique(y_encoded))


# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.5,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

