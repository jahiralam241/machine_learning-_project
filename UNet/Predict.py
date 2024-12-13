import os
import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from keras.utils import CustomObjectScope
from Data import load_data, tf_dataset
from Train import iou  # Use the updated IoU function

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Function to read images
def read_image(PATH):
    x = cv2.imread(PATH, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (128, 128))
    x = x / 255.0
    return x

# Function to read masks
def read_mask(PATH):
    x = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (128, 128))
    x = np.expand_dims(x, axis=-1)
    return x

# Function to parse masks for visualization
def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # Path to the dataset
    PATH = "/home/jahir/Desktop/project/Dataset/"
    BATCH = 8

    # Load the dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(PATH)

    # Create a dataset for testing
    test_dataset = tf_dataset(test_x, test_y, batch=BATCH)
    test_steps = len(test_x) // BATCH

    # Adjust test steps if needed
    if len(test_x) % BATCH != 0:
        test_steps += 1

    # Load the saved model and include the IoU metric
    with CustomObjectScope({'iou': iou}):  # Import the updated IoU metric from Train.py
        model = tf.keras.models.load_model("model.keras")  # Load the .keras file

    # Evaluate the model on the test set
    model.evaluate(test_dataset, steps=test_steps)

    # Loop through the test dataset and make predictions
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)  # Read the input image
        y = read_mask(y)   # Read the true mask

        # Make a prediction
        y_pred = model.predict(np.expand_dims(x, axis=0))
        y_pred = y_pred[0] > 0.5  # Convert prediction to binary mask

        # Prepare visualization by stacking input image, true mask, and predicted mask
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0
        all_images = [
            x * 255.0, white_line,  # Input image
            mask_parse(y), white_line,  # True mask
            mask_parse(y_pred) * 255.0  # Predicted mask
        ]
        image = np.concatenate(all_images, axis=1)  # Concatenate for side-by-side view
        cv2.imwrite(f"hhome/jahir/Desktop/project/Results_unet/{i}.png", image)

        # Save the result image
        cv2.imwrite(f"/home/jahir/Desktop/project/Results_unet/{i}.png", image)

