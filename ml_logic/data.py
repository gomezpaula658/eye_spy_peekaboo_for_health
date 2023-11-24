from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
import numpy as np

def initialize_model():
    """
    Initialize the Neural Network with random weights
    """
    pass

    #print("✅ Model initialized")

    #return model

def compile_model():
    """
    Compile the Neural Network
    """
    pass

    #print("✅ Model compiled")

    #return model

def train_model():
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    pass

    #print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    #return model, history
