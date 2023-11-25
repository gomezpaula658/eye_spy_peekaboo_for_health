import os
import numpy as np
from tensorflow import keras
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
from keras import Model, Sequential, layers
import pandas as pd
from keras.preprocessing import image

def load_and_preprocess_image(uploaded_image):
    """
    Load and preprocess an image given its ID.
    """
    image = Image.open(uploaded_image)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Create a function to balance the data.
def data_balancing(table_link='../Data/'):
    '''
    This function loads the raw data and returns a balanced dataset as two
    dictionaries.
    '''
    df = pd.read_csv(f'{table_link}RFMiD_Training_Labels.csv')
    df_binary = df.loc[:, df.columns.intersection(['ID','Disease_Risk'])]
    df_healthy = df_binary[df.Disease_Risk == 0]
    df_unhealthy = df_binary[df.Disease_Risk == 1]
    df_unhealthy_rndselection = df_unhealthy.sample(n = 401)
    frames = [df_healthy, df_unhealthy_rndselection]
    result = pd.concat(frames)
    return result

def load_and_preprocess_image(image_id, image_folder, target_size=(224, 224)):
    """
    Load and preprocess an image given its ID.
    """
    image_path = os.path.join(image_folder, f'{image_id}.png')
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0
    return image

def create_augmented_model(input_shape=(150, 150, 3)):

    """
    Image augmnetation function Layers
    """

    data_augmentation = Sequential()
    data_augmentation.add(layers.Rescaling(1./255, input_shape=input_shape))
    data_augmentation.add(layers.RandomFlip("horizontal"))
    data_augmentation.add(layers.RandomZoom(0.1))
    data_augmentation.add(layers.RandomTranslation(0.2, 0.2))
    data_augmentation.add(layers.RandomRotation(0.1))

    return data_augmentation
