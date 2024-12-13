from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model

def build_vgg16_unet(input_shape=(128, 128, 3)):
    # Load VGG-16 without the top fully connected layers
    vgg16 = VGG16(weights=None, include_top=False, input_shape=input_shape)

    # Encoder: Extract the feature maps from specific layers in VGG-16
    inputs = vgg16.input
    skip1 = vgg16.get_layer("block1_conv2").output  # Shape: (None, 128, 128, 64)
    skip2 = vgg16.get_layer("block2_conv2").output  # Shape: (None, 64, 64, 128)
    skip3 = vgg16.get_layer("block3_conv3").output  # Shape: (None, 32, 32, 256)
    skip4 = vgg16.get_layer("block4_conv3").output  # Shape: (None, 16, 16, 512)

    # Bridge (lowest layer)
    bridge = vgg16.get_layer("block5_conv3").output  # Shape: (None, 8, 8, 512)

    # Decoder: Use upsampling layers and concatenate skip connections
    x = UpSampling2D((2, 2))(bridge)  # Upsample from 8x8 to 16x16
    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = Concatenate()([x, skip4])  # Concatenate skip connection from encoder

    x = UpSampling2D((2, 2))(x)  # Upsample from 16x16 to 32x32
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = Concatenate()([x, skip3])

    x = UpSampling2D((2, 2))(x)  # Upsample from 32x32 to 64x64
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = Concatenate()([x, skip2])

    x = UpSampling2D((2, 2))(x)  # Upsample from 64x64 to 128x128
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = Concatenate()([x, skip1])

    # Final upsample to the original image size (128x128)
   
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)

    # Output layer: Binary mask with 1 channel (use sigmoid for binary segmentation)
    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)

    # Create the model
    model = Model(inputs, output)
    
    return model
if __name__ == "__main__":
    model = build_vgg16_unet()
    model.summary()
