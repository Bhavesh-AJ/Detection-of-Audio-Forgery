import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_metadata(dataset_path):
    """
    dataset_path = path containing meta.csv and wav files
    """

    meta_path = os.path.join(dataset_path, "meta.csv")

    df = pd.read_csv(meta_path)

    # Convert labels
    df["label"] = df["label"].map({
        "bona-fide": 0,
        "spoof": 1
    })

    # Full audio path
    df["audio_path"] = df["file"].apply(
        lambda x: os.path.join(dataset_path, x)
    )

    return df


def split_data(df, test_size=0.2):

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )

    return train_df, test_df


if __name__ == "__main__":

    DATASET_PATH = r"Data\release_in_the_wild"

    df = load_metadata(DATASET_PATH)

    train_df, test_df = split_data(df)

    print("Total Samples :", len(df))
    print("Train Samples :", len(train_df))
    print("Test Samples  :", len(test_df))

    print(df.head())