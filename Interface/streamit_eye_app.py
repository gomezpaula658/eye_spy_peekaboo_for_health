import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from PIL import Image
from ml_logic.preprocessor import load_and_preprocess_image


## Front page of the app displaying a form for the doctor to enter the patient name
## and upload their eye scan image. The interface then return the result i.e healthy/not Healthy.
## This function will return the downloaded image.

#read css file
# with open("style.css") as f:
#     st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)


def predict(image_processed):
    """Display the result"""
    prediction_result = model.predict(image_processed)
    #getting class index with highest probability 
    first_probability, second_probability = prediction_result
    if first_probability > second_probability:
        return "Healthy"
    else:
        return "Unhealthy"
    #return result from the model
    #load trained model saved
    #call built-in predict funtion on model
    #return result that has bigger prob


with st.container():
    st.markdown("<center><h1>Eye Spy</h1></center>", unsafe_allow_html=True)

    patient = st.text_input('Enter Patient Name', '', key="patient_name")
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
    # To read file as bytes:
        #bytes_data = uploaded_file.getvalue()
        image = load_and_preprocess_image(uploaded_file)

    submitted = st.button("Submit", key="submit_button")

    if submitted:
        st.image(uploaded_file, caption=f"{patient}'s eye image")
        #code to show processing
        with st.spinner('Classifying...'):
            #perform prediction
            prediction_result = predict(image)
            
        #Display prediction result
        st.write(f"Precition: {prediction_result}")
        
        #st.markdown(f"<center><p>{st.write('Healthy')}</p></center>", unsafe_allow_html=True)
        # st.write("Classifying...")
        #call predict function from model
        # label = predict(image)
        # st.write('%s (%.2f%%)' % (label[1], label[2]*100))
