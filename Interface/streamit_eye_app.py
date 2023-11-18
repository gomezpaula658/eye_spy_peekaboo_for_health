import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

def eye_interface():
    """ Front page of the app displaying a form for the doctor to enter the patient name
    and upload their eye scan image. The interface then return the result i.e healthy/not Healthy.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Column 1")

    with col2:

    with st.form("main_form"):
        st.title('Eye Spy')

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





        with col2:
            st.markdown("<center><button>Click me!</button></center>", unsafe_allow_html=True)


        submitted = st.form_submit_button("Submit")
        if submitted:
            st.image(uploaded_file, caption=f"{patient}'s eye image")
            clicked = st.button("Get results", key="result_button")
            if clicked:
                st.write('Hello, *World!* :sunglasses:')

    return uploaded_file

def get_results():
    """ Get the result from the model for the downloaded image"""

    # def write():
    #     st.write('Hello, *World!* :sunglasses:')

    # st.form_submit_buttonbutton("Get results", key="result_button", on_click=write)
        ### CODE HERE TO GET THE RESULT FROM MODEL


eye_interface()
