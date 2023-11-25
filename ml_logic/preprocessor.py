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


def create_augmented_model(input_shape=(150, 150, 3)):
    
    """
    Image augmnetation function Layers
    """
    
    model.add(layers.Rescaling(1./255, input_shape=input_shape))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomZoom(0.1))
    model.add(layers.RandomTranslation(0.2, 0.2))
    model.add(layers.RandomRotation(0.1))
    
    return model