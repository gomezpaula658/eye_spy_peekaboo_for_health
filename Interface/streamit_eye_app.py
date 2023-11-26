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
train = pd.read_csv('../data/RFMiD_Training_Labels.csv').set_index('ID')
test = pd.read_csv('../data/RFMiD_Testing_Labels.csv').set_index('ID')
eval = pd.read_csv('../data/RFMiD_Validation_Labels.csv').set_index('ID')

X_train = train.drop(columns='Disease_Risk')
y_train = train['Disease_Risk']
X_eval  = eval.drop(columns='Disease_Risk')
y_eval = eval['Disease_Risk']

image_folder = '../data/training_images'
images = np.array([load_and_preprocess_images(row_id, image_folder) for row_id in X_train.index])
eval_image_folder = '../data/eval_images'
eval_images = np.array([load_and_preprocess_images(row_id, image_folder) for row_id in X_eval.index])

if not local_model_paths:
    model = initialize_model((224, 224, 3))
    model = compile_model(model)
    model = train_model(images, y_train, validation_data=(eval_images, y_eval))
    save_model(model)



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
        image = load_and_preprocess_image_uploaded(uploaded_file)

    submitted = st.button("Submit", key="submit_button")

    if submitted:
        st.image(uploaded_file, caption=f"{patient}'s eye image")
        #code to show processing
        with st.spinner('Classifying...'):
            #perform prediction
            prediction_result = predict(image)

        #Display prediction result
        st.write(f"Prediction: {prediction_result}")

        #st.markdown(f"<center><p>{st.write('Healthy')}</p></center>", unsafe_allow_html=True)
        # st.write("Classifying...")
        #call predict function from model
        # label = predict(image)
        # st.write('%s (%.2f%%)' % (label[1], label[2]*100))
