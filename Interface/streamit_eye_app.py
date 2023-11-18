import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

def eye_interface():
    """ Front page of the app displaying a form for the doctor to enter the patient name
    and upload their eye scan image. The interface then return the result i.e healthy/not Healthy.
    """
    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.write("Column 1")

    # with col2:
    #     st.title('CLAIRE\'S WORLD')
    #with st.form("main_form"):
        # st.title('Eye Spy')
    st.markdown("<center><h1>Eye Spy</h1></center>", unsafe_allow_html=True)


    #st.text_area("Patient Name", value="Enter the patient name:")
    patient = st.text_input('Enter Patient Name', '', key="patient_name")

    uploaded_file = st.file_uploader("Choose a CSV file")
    # for uploaded_file in uploaded_files:
    #     bytes_data = uploaded_file.read()
    #     st.write("filename:", uploaded_file.name)
    #     st.write(bytes_data)
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        #st.write(bytes_data)
        # To convert to a string based IO:
        # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)

        # # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)

        # # Can be used wherever a "file-like" object is accepted:
        # dataframe = pd.read_csv(uploaded_file)
        # st.write(dataframe)

    submitted = st.button("Submit", key="submit_button")
    if submitted:
        with st.form("main_form"):
            st.image(uploaded_file, caption=f"{patient}'s eye image")
            clicked = st.form_submit_button("Get results")
            if clicked:
                st.write(get_results())

    # with col3:
    #     st.write('Did this work?')
    return uploaded_file


def get_results():
    """ Get the result from the model for the downloaded image"""
    def write():
        st.write('Hello, *World!* :sunglasses:')
    return write()


eye_interface()


# RANDOM MARKDOWN CODE:
        # with col2:
            # st.markdown("<center><button>Click me!</button></center>", unsafe_allow_html=True)
