import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from PIL import Image

def load_and_preprocess_image(uploaded_image):
    """
    Load and preprocess an image given its ID.
    """
    image = Image.open(uploaded_image)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image
