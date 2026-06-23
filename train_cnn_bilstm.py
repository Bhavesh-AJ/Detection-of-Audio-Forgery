import math
import time
import numpy as np
import tensorflow as tf

from dataset_loader import load_metadata, split_data
from feature_extraction import extract_mfcc_for_dl
from cnn_bilstm_model import build_cnn_bilstm




DATASET_PATH = r"Data\release_in_the_wild"

BATCH_SIZE = 32
EPOCHS = 5




print("Loading metadata...")

df = load_metadata(DATASET_PATH)

train_df, test_df = split_data(df)

print("\nDataset Information")
print("Train Samples:", len(train_df))
print("Test Samples :", len(test_df))




class TimeHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.training_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):

        epoch_time = time.time() - self.epoch_start
        total_time = time.time() - self.training_start

        print(
            f"\nEpoch {epoch + 1} completed "
            f"in {epoch_time:.2f} seconds"
        )

        print(
            f"Total elapsed time: "
            f"{total_time / 60:.2f} minutes\n"
        )




class AudioGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataframe, batch_size=32):

        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size

    def __len__(self):

        return math.ceil(
            len(self.df) / self.batch_size
        )

    def __getitem__(self, idx):

        batch_df = self.df.iloc[
            idx * self.batch_size:
            (idx + 1) * self.batch_size
        ]

        X = []
        y = []

        for _, row in batch_df.iterrows():

            try:

                mfcc = extract_mfcc_for_dl(
                    row["audio_path"]
                )

                X.append(mfcc)
                y.append(row["label"])

            except Exception as e:

                print(
                    f"Error processing: "
                    f"{row['audio_path']}"
                )

                print(e)

        X = np.array(
            X,
            dtype=np.float32
        )

        y = np.array(y)

        return X, y




print("\nCreating generators...")

train_generator = AudioGenerator(
    train_df,
    batch_size=BATCH_SIZE
)

test_generator = AudioGenerator(
    test_df,
    batch_size=BATCH_SIZE
)

print(
    f"Train Batches: "
    f"{len(train_generator)}"
)

print(
    f"Test Batches : "
    f"{len(test_generator)}"
)




print("\nBuilding CNN-BiLSTM model...")

model = build_cnn_bilstm()

model.summary()




print("\nTraining Started...\n")

timer = TimeHistory()

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[timer],
    verbose=1
)




print("\nSaving model...")

model.save(
    "cnn_bilstm_full.keras"
)

print(
    "\nModel saved as "
    "'cnn_bilstm_full.keras'"
)

print("\nTraining Complete.")