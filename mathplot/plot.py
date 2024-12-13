import pickle
import matplotlib.pyplot as plt

# Load the history of Model 1
with open('/home/jahir/Desktop/project/UNet/model_1_history.pkl', 'rb') as f:
    history_1 = pickle.load(f)

# Load the history of Model 2
with open('/home/jahir/Desktop/project/vgg-16/model_2_history.pkl', 'rb') as f:
    history_2 = pickle.load(f)

# Plot Training and Validation Metrics for both models
plt.figure(figsize=(18, 6))

# Accuracy Comparison
plt.subplot(1, 3, 1)  # Plot accuracy on the left
plt.plot(history_1['accuracy'], label='unet - Train Accuracy', color='blue', linestyle='-')

plt.plot(history_2['accuracy'], label='vgg-16 - Train Accuracy', color='green', linestyle='-')

plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Comparison
plt.subplot(1, 3, 2)  # Plot loss in the middle
plt.plot(history_1['loss'], label='unet - Train Loss', color='red', linestyle='-')

plt.plot(history_2['loss'], label='vgg-16 - Train Loss', color='orange', linestyle='-')

plt.title('Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# IoU Comparison
plt.subplot(1, 3, 3)  # Plot IoU on the right
plt.plot(history_1['iou'], label='unet - Train IoU', color='purple', linestyle='-')

plt.plot(history_2['iou'], label='vgg-16 - Train IoU', color='cyan', linestyle='-')

plt.title('IoU Comparison')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()

