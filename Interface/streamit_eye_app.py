import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

def eye_interface():

    with st.form("my_form"):
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

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.image(uploaded_file, caption=patient)

    #st.button("Get results", key="result_button", disabled=True)

    return uploaded_file

eye_interface()
