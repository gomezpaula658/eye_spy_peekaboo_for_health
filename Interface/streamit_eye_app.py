import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

def eye_interface():
    """ Front page of the app displaying a form for the doctor to enter the patient name
    and upload their eye scan image. The interface then return the result i.e healthy/not Healthy.
    This function will return the downloaded image.
    """
    #read css file
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)

    with st.container():
        st.markdown("<center><h1>Eye Spy</h1></center>", unsafe_allow_html=True)

        patient = st.text_input('Enter Patient Name', '', key="patient_name")
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
        # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

        st.button("Submit", key="submit_button", on_click=display_eye, args=[uploaded_file,patient])

    return uploaded_file


def display_results():
    """Display the result"""
    #Tempory function to test interface behaviour
    def write():
        st.write('Healthy')
    return write()

def display_eye(file, patient):
    """Display the downloaded image and the name of the patient"""
    st.image(file, caption=f"{patient}'s eye image")
    st.button("Get results", key="result_button", on_click=display_results)

def get_result():
    """ Get the result from the model for the downloaded image"""
    pass ##CONNECTION WITH MODEL

#Main function call
eye_interface()
