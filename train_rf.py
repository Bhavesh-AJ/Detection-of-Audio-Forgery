import numpy as np
import joblib
import matplotlib.pyplot as plt

from dataset_loader import load_metadata, split_data
from feature_extraction import extract_chroma
from random_forest_model import build_random_forest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

DATASET_PATH = r"Data\release_in_the_wild"

print("Loading metadata...")

df = load_metadata(DATASET_PATH)

train_df, test_df = split_data(df)

print("\nDataset Information")
print("Train Samples:", len(train_df))
print("Test Samples :", len(test_df))



print("\nExtracting Chroma Features...")

X_train = []
y_train = []

for _, row in train_df.iterrows():

    try:

        feature = extract_chroma(
            row["audio_path"]
        )

        X_train.append(feature)
        y_train.append(row["label"])

    except Exception as e:
        print(e)

X_test = []
y_test = []

for _, row in test_df.iterrows():

    try:

        feature = extract_chroma(
            row["audio_path"]
        )

        X_test.append(feature)
        y_test.append(row["label"])

    except Exception as e:
        print(e)

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

print("\nDataset Shapes")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)



model = build_random_forest()

print("\nTraining Random Forest...")

model.fit(X_train, y_train)



print("Generating predictions...")

preds = model.predict(X_test)



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

print("\n========== RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report")
print(
    classification_report(
        y_test,
        preds,
        target_names=[
            "Bona-Fide",
            "Spoof"
        ]
    )
)



cm = confusion_matrix(
    y_test,
    preds
)

print("\nConfusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Bona-Fide",
        "Spoof"
    ]
)

disp.plot()

plt.title(
    "Random Forest Confusion Matrix"
)

plt.savefig(
    "rf_confusion_matrix.png"
)

plt.close()



joblib.dump(
    model,
    "random_forest.pkl"
)

print("\nModel saved as random_forest.pkl")
print("Confusion matrix saved as rf_confusion_matrix.png")