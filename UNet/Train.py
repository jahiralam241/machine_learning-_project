import os
import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)
from tensorflow.keras.metrics import Recall, Precision
from Data import load_data, tf_dataset
from Model import build_model

# Suppress TensorFlow warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Custom IoU metric using pure TensorFlow operations
def iou(y_true, y_pred):
    # Cast y_true and y_pred to float32 to avoid data type mismatches
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Flatten the tensors
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    # Compute IoU
    iou_value = (intersection + 1e-15) / (union + 1e-15)
    return iou_value


if __name__ == "__main__":
    # Path to dataset
    PATH = "/home/jahir/Desktop/project/Dataset/"
    
    # Load dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(PATH)

    # Define batch size, learning rate, and number of epochs
    BATCH = 8
    lr = 1e-4
    epochs = 10

    # Create datasets using tf_dataset function
    train_dataset = tf_dataset(train_x, train_y, batch=BATCH)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)

    # Build the U-Net model
    model = build_model()

    # Compile the model
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["accuracy", Recall(), Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    # Set up callbacks for saving model and adjusting learning rate
    callbacks = [
        ModelCheckpoint("model.keras"),  # Use .keras extension as required
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
        CSVLogger("data.csv"),
        TensorBoard(),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=False
        )
    ]

    # Determine number of steps per epoch for training and validation
    train_steps = len(train_x) // BATCH
    valid_steps = len(valid_x) // BATCH

    # Adjust steps per epoch if dataset size is not divisible by batch size
    if len(train_x) % BATCH != 0:
        train_steps += 1
    if len(valid_x) % BATCH != 0:
        valid_steps += 1

    # Train the model with the dataset
    history_1=model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )
    with open('model_1_history.pkl', 'wb') as f:
    	pickle.dump(history_1.history, f)

