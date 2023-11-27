import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
from io import StringIO
from PIL import Image
from ml_logic.preprocessor import load_and_preprocess_image_uploaded, load_and_preprocess_images
from ml_logic.registry import load_model
from params import *
from ml_logic.model import *
from ml_logic.registry import *


## Front page of the app displaying a form for the doctor to enter the patient names
## and upload their eye scan image. The interface then return the result i.e healthy/not Healthy.
## This function will return the downloaded image.

#read css file
# with open("style.css") as f:
#     st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)
local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
local_model_paths = glob.glob(f"{local_model_directory}/*")


#Read the files and get the images to train the model
# train = pd.read_csv(f'{LOCAL_DATA_PATH1}/RFMiD_Training_Labels.csv').set_index('ID')
# test = pd.read_csv(f'{LOCAL_DATA_PATH1}/RFMiD_Testing_Labels.csv').set_index('ID')
# eval = pd.read_csv(f'{LOCAL_DATA_PATH1}/RFMiD_Validation_Labels.csv').set_index('ID')

# X_train = train.drop(columns='Disease_Risk')
# y_train = train['Disease_Risk']
# X_eval  = eval.drop(columns='Disease_Risk')
# y_eval = eval['Disease_Risk']

# image_folder = f'{LOCAL_DATA_PATH1}/training_images'
# images = np.array([load_and_preprocess_images(row_id, image_folder) for row_id in X_train.index])
# eval_image_folder = f'{LOCAL_DATA_PATH1}/eval_images'
# eval_images = np.array([load_and_preprocess_images(row_id, image_folder) for row_id in X_eval.index])

# #if not local_model_paths:
    # learning_rate = 0.0005
    # batch_size = 256
    # patience = 2
    # model = initialize_model((224, 224, 3))
    # model = compile_model(model)
    # model, history = train_model(model, images, y_train, batch_size=batch_size, patience=patience, validation_data=(eval_images, y_eval), validation_split=None)
    # save_model(model)



def predict(image_processed):
    """Display the result"""
    #checking that there is a model saved
    if not local_model_paths:
        return None
    else:
        model = load_model()
        prediction_result = model.predict(image_processed)
        #getting class index with highest probability
        first_probability, second_probability = prediction_result
        if first_probability > second_probability:
            return "Healthy"
        else:
            return "Unhealthy"


with st.container():
    st.markdown("<center><h1>Eye Spy</h1></center>", unsafe_allow_html=True)

    patient = st.text_input('Enter Patient Name', '', key="patient_name")
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
    # To read file as bytes:
        #bytes_data = uploaded_file.getvalue()
        image = np.array(load_and_preprocess_images(99,f'{LOCAL_DATA_PATH1}/test_images'))

    submitted = st.button("Submit", key="submit_button")

    if submitted:
        st.image(uploaded_file, caption=f"{patient}'s eye image")
        #code to show processing
        with st.spinner('Classifying...'):
            #perform prediction
            prediction_result = predict(image)

        #Display prediction result
        st.write(f"Prediction: {prediction_result}")
