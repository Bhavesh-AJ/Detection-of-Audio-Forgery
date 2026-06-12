import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from dataset_loader import load_metadata, split_data
from feature_extraction import extract_mfcc_for_dl


# --------------------------------
# Configuration
# --------------------------------

DATASET_PATH = r"Data\release_in_the_wild"
MODEL_PATH = "cnn_bilstm_full.keras"

os.makedirs("results", exist_ok=True)

# --------------------------------
# Load Dataset
# --------------------------------

print("Loading metadata...")

df = load_metadata(DATASET_PATH)

train_df, test_df = split_data(df)

# Same subset used during training
test_df = test_df.head(1000)

# --------------------------------
# Feature Extraction
# --------------------------------

X_test = []
y_test = []

print("Extracting MFCC features...")

for _, row in test_df.iterrows():

    try:

        mfcc = extract_mfcc_for_dl(
            row["audio_path"]
        )

        X_test.append(mfcc)
        y_test.append(row["label"])

    except Exception as e:

        print(
            f"Error processing {row['audio_path']}"
        )

        print(e)

X_test = np.array(
    X_test,
    dtype=np.float32
)

y_test = np.array(y_test)

print("\nDataset Shapes")

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# --------------------------------
# Load Saved Model
# --------------------------------

print("\nLoading CNN-BiLSTM model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

# --------------------------------
# Prediction
# --------------------------------

print("Generating predictions...")

pred_probs = model.predict(
    X_test
)

preds = (
    pred_probs > 0.5
).astype(int)

preds = preds.flatten()

# --------------------------------
# Metrics
# --------------------------------

accuracy = accuracy_score(
    y_test,
    preds
)

precision = precision_score(
    y_test,
    preds
)

recall = recall_score(
    y_test,
    preds
)

f1 = f1_score(
    y_test,
    preds
)

roc_auc = roc_auc_score(
    y_test,
    pred_probs
)

print("\n========== RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# --------------------------------
# Confusion Matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Bona-Fide",
        "Spoof"
    ]
)

disp.plot()

plt.title(
    "CNN-BiLSTM Confusion Matrix"
)

plt.savefig(
    "results/confusion_matrix.png"
)

plt.close()

# --------------------------------
# ROC Curve
# --------------------------------

fpr, tpr, _ = roc_curve(
    y_test,
    pred_probs
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.savefig(
    "results/roc_curve.png"
)

plt.close()

print("\nResults saved successfully.")

print("results/confusion_matrix.png")
print("results/roc_curve.png")