import os
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras import Model, Sequential, layers, models
from tensorflow.keras.preprocessing import image
from PIL import Image

def load_and_preprocess_image_uploaded(uploaded_image):
    """
    Load and preprocess an image given its ID.
    """
    image = Image.open(uploaded_image)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_images(image_id, image_folder, target_size=(224, 224)):
    """
    Load and preprocess an image given its ID.
    """
    image_path = os.path.join(image_folder, f'{image_id}.png')
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0
    return image

# Create a function to balance the data.
def data_balancing(table_link='../data/'):
    '''
    This function loads the raw data and returns a balanced dataset as two
    dictionaries.
    '''
    df = pd.read_csv(f'{table_link}RFMiD_Training_Labels.csv').set_index('ID')
    df_binary = df.loc[:, df.columns.intersection(['Disease_Risk'])]
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

def image_augmentation(images):

    data_augmentation = models.Sequential([
    # layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.001),
    layers.RandomTranslation(0.02, 0.02),
    layers.RandomRotation(0.1)
    ])

    # images = tf.expand_dims(images, 0)

    augmented_images = data_augmentation(images)

    # images = tf.squeeze(images, 0)

    return augmented_images

def create_model(shape=tuple):

    """
    Image augmnetation function Layers
    """

    image_input = layers.Input(shape=shape)

    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Flatten()(x)

    z = layers.Dense(12, activation='relu')(x)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dense(1, activation='sigmoid')(z)

    model = Model(inputs=image_input, outputs=z)

    return model
