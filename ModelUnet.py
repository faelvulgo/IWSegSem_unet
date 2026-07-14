import torch
import segmentation_models_pytorch as smp

#Define the encoder.
ENCODER = "resnet34"

# Define the weights for the encoder.
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "softmax"

# Create the model using the specified encoder and weights.
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    activation=ACTIVATION
)

# Normalize data.
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

print(model)