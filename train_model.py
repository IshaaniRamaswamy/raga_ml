from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from dataset_loader import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Load dataset
X, y = load_dataset("dataset")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.5,
    random_state=42,
    stratify=y_encoded
)

# Train SVM model
model = SVC(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

for true, pred in zip(y_test, y_pred):
    print("True:", encoder.inverse_transform([true])[0],
          "| Pred:", encoder.inverse_transform([pred])[0])

