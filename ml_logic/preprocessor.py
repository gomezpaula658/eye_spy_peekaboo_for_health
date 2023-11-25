import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras import Model, Sequential, layers
import pandas as pd

def load_and_preprocess_image(uploaded_image):
    """
    Load and preprocess an image given its ID.
    """
    image = Image.open(uploaded_image)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Create a function to balance the data.
def data_balancing():
    '''
    This function loads the raw data and returns a balanced dataset as two
    dictionaries.
    '''
    # Load data.
    df = pd.read_csv('../raw_data/RFMiD_Training_Labels.csv')
    # Remove disease categories, only keep binary healthy or unhealty column.
    df_binary = df.loc[:, df.columns.intersection(['ID','Disease_Risk'])]
    # Define healthy and unhealthy data sets.
    df_healthy = df_binary[df.Disease_Risk == 0]
    df_unhealthy = df_binary[df.Disease_Risk == 1]
    # Select 401 random samples from unhealthy data set.
    df_unhealthy_rndselection = df_unhealthy.sample(n = 401)
    # Create a dictionary to pair ID's with images for each data set.
    img_dict_healthy = {}
    for id_no in df_healthy.ID:
        img = Image.open(f"../raw_data/Training/{id_no}.png")
        img_dict_healthy[id_no] = img
    img_dict_unhealthy = {}
    for id_no in df_unhealthy_rndselection.ID:
        img = Image.open(f"../raw_data/Training/{id_no}.png")
        img_dict_unhealthy[id_no] = img
    # Return both dictionaries.
    return (img_dict_healthy, img_dict_unhealthy)
  

def create_augmented_model(input_shape=(150, 150, 3)):

    """
    Image augmnetation function Layers
    """

    model = Sequential()
    model.add(layers.Rescaling(1./255, input_shape=input_shape))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomZoom(0.1))
    model.add(layers.RandomTranslation(0.2, 0.2))
    model.add(layers.RandomRotation(0.1))

    return model

