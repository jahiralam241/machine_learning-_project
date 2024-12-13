import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Step 1: Define the custom IoU function
def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou_value = (intersection + 1e-15) / (union + 1e-15)
    return iou_value

# Step 2: Load the models and include the custom IoU metric during loading
def load_models(unet_path, vgg16_path):
    unet_model = load_model(unet_path, custom_objects={'iou': iou})
    vgg16_model = load_model(vgg16_path, custom_objects={'iou': iou})
    return unet_model, vgg16_model

# Load models
unet_model, vgg16_model = load_models('/home/jahir/Desktop/project/UNet/model.keras',
                                      '/home/jahir/Desktop/project/vgg-16/build_vgg16_unet.keras')

# Step 3: Load and preprocess the dataset from separate directories
def load_images_and_masks(image_directory, mask_directory, target_size):
    images = []
    masks = []
    
    image_filenames = [f for f in os.listdir(image_directory) if f.endswith(".png")]
    mask_filenames = [f for f in os.listdir(mask_directory) if f.endswith(".png")]
    
    # Sort filenames to ensure proper pairing
    image_filenames.sort()
    mask_filenames.sort()
    
    for img_filename in image_filenames:
        img_path = os.path.join(image_directory, img_filename)
        mask_filename = img_filename.replace('Image', 'Mask')  # Adjust if masks follow a different naming convention
        mask_path = os.path.join(mask_directory, mask_filename)
        
        # Load image
        if os.path.isfile(img_path):
            image = load_img(img_path, target_size=target_size, color_mode='rgb')  # Ensure correct color mode
            image = img_to_array(image) / 255.0  # Normalize
            images.append(image)
        else:
            print(f"Warning: Image file {img_path} not found.")
        
        # Load mask
        if os.path.isfile(mask_path):
            mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')  # Use 'grayscale' for masks
            mask = img_to_array(mask) / 255.0  # Normalize
            masks.append(mask)
        else:
            print(f"Warning: Mask file {mask_path} not found.")
    
    return np.array(images), np.array(masks)

# Define directories and target size
image_dir = "/home/jahir/Desktop/project/Dataset/UNET-Data/Transversal"
mask_dir = "/home/jahir/Desktop/project/Dataset/UNET-Data/Transversal-Mask"
target_size = (128, 128)  # Adjust to match the input size expected by your models

# Load dataset
X_test, y_test = load_images_and_masks(image_dir, mask_dir, target_size)

# Ensure masks are binary
y_test = (y_test > 0.5).astype(np.float32)

# Predict probabilities using the U-Net and VGG-16 models
y_pred_unet = unet_model.predict(X_test)
y_pred_vgg16 = vgg16_model.predict(X_test)

# Convert predictions to probabilities if needed
y_pred_unet = tf.keras.activations.sigmoid(y_pred_unet).numpy()
y_pred_vgg16 = tf.keras.activations.sigmoid(y_pred_vgg16).numpy()

# Flatten the predictions and ground truth masks to prepare for ROC analysis
y_pred_unet_flat = y_pred_unet.ravel()
y_pred_vgg16_flat = y_pred_vgg16.ravel()
y_true_flat = y_test.ravel()

# Ensure all arrays are properly shaped and check for NaNs
if np.any(np.isnan(y_true_flat)) or np.any(np.isnan(y_pred_unet_flat)) or np.any(np.isnan(y_pred_vgg16_flat)):
    raise ValueError("NaN values found in the data. Please check the inputs.")

# Calculate the ROC curve and AUC for both models
fpr_unet, tpr_unet, _ = roc_curve(y_true_flat, y_pred_unet_flat)
auc_unet = auc(fpr_unet, tpr_unet)

fpr_vgg16, tpr_vgg16, _ = roc_curve(y_true_flat, y_pred_vgg16_flat)
auc_vgg16 = auc(fpr_vgg16, tpr_vgg16)

# Plot the ROC curves for both models
plt.figure()

plt.plot(fpr_unet, tpr_unet, color='blue', label=f'U-Net (AUC = {auc_unet:.2f})')
plt.plot(fpr_vgg16, tpr_vgg16, color='green', label=f'VGG-16 (AUC = {auc_vgg16:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: U-Net vs VGG-16')
plt.legend(loc='lower right')

plt.show()

