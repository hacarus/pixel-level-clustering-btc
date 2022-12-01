"""UNet."""
from typing import Optional
from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras.models import Model 
from tensorflow.keras.applications import MobileNetV2 


def unet(size: int, weight_path: Optional[Path] = None) -> Model:
    """Return UNet model.

    Parameters
    ----------
        size: int,
            patch size.
        weight_path: Optional[Path],
            path to weight.
    
    Returns
    -------
        model: Model
    """
    inputs = layers.Input(shape=(size, size, 3), name="input_image")
    feature_extractor = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    feature_out = feature_extractor.get_layer("block_13_expand_relu").output
    
    layer_size = [16, 32, 48, 64]
    x = feature_out 
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = feature_extractor.get_layer(skip_connection_names[-i]).output 
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, x_skip])
        x = layers.Activation("relu")(x)

        for _ in range(2): 
            x = layers.Conv2D(layer_size[-i], (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        
    x = layers.Conv2D(1, (1, 1), padding="same")(x)
    x = layers.Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    if weight_path:
        model.load_weights(weight_path) 
    return model
