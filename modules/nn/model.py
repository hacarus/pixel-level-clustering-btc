"""UNet."""
from typing import Callable, Optional
from pathlib import Path
from tensorflow.keras.models import Model 


def get_model(model: Callable, size: int, weight_path: Optional[Path] = None) -> Model:
    return model(size, weight_path)
