import tensorflow as tf


def build_cnn_bilstm():

    model = tf.keras.Sequential()


    model.add(
        tf.keras.layers.Input(
            shape=(300, 40)
        )
    )

    
    model.add(
        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        )
    )

    model.add(
        tf.keras.layers.MaxPooling1D(
            pool_size=2
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization()
    )

    
    model.add(
        tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu"
        )
    )

    model.add(
        tf.keras.layers.MaxPooling1D(
            pool_size=2
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization()
    )

    
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                64,
                return_sequences=False
            )
        )
    )

    
    model.add(
        tf.keras.layers.Dropout(
            0.3
        )
    )

    
    model.add(
        tf.keras.layers.Dense(
            64,
            activation="relu"
        )
    )

    
    model.add(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_cnn_bilstm()

    model.summary()