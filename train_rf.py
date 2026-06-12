import numpy as np
import joblib

from dataset_loader import load_metadata, split_data
from feature_extraction import extract_chroma
from random_forest_model import build_random_forest

from sklearn.metrics import accuracy_score


DATASET_PATH = r"Data\release_in_the_wild"

print("Loading metadata...")

df = load_metadata(DATASET_PATH)

train_df, test_df = split_data(df)

# --------------------------
# Limit samples initially
# --------------------------

train_df = train_df.head(2000)
test_df = test_df.head(500)

print("Extracting Chroma Features...")

X_train = []
y_train = []

for _, row in train_df.iterrows():

    try:

        feature = extract_chroma(
            row["audio_path"]
        )

        X_train.append(feature)

        y_train.append(
            row["label"]
        )

    except Exception:
        pass


X_test = []
y_test = []

for _, row in test_df.iterrows():

    try:

        feature = extract_chroma(
            row["audio_path"]
        )

        X_test.append(feature)

        y_test.append(
            row["label"]
        )

    except Exception:
        pass


X_train = np.array(X_train)
X_test = np.array(X_test)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# --------------------------
# Train RF
# --------------------------

model = build_random_forest()

print("Training Random Forest...")

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(
    y_test,
    preds
)

print("\nAccuracy:", acc)

joblib.dump(
    model,
    "random_forest.pkl"
)

print("Model saved.")
