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
from keras.utils import load_img, img_to_array
from tensorflow import keras
from tempfile import NamedTemporaryFile


## Front page of the app displaying a form for the doctor to enter the patient names
## and upload their eye scan image. The interface then return the result i.e healthy/not Healthy.
## This function will return the downloaded image.


def pred(image_processed):
    """Display the result"""
    model = keras.models.load_model("models/model_1.h5")
    image_processed = image_processed.reshape((1,224,224,3))
    prediction_result = model.predict(image_processed)
    #getting class index with highest probability
    st.write(prediction_result)
    print(prediction_result)
    if prediction_result > 0.50:
        return "Unhealthy"
    else:
        return "Healthy"

#stage 2 - predict disease if eye is unhealthy
def pred2(image_processed):
    model1 = keras.models.load_model("models/model_2.h5")
    image_processed = image_processed.reshape((1,224,224,3))
    prediction_result1 = model1.predict(image_processed)
    #getting class index with highest probability
    st.write(prediction_result1)
    return prediction_result1

with st.container():
    st.markdown("<center><h1>Eye Spy</h1></center>", unsafe_allow_html=True)

    patient = st.text_input('Enter Patient Name', '', key="patient_name")
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
    # To read file as bytes:
        #bytes_data = uploaded_file.getvalue()
        #image = load_and_preprocess_images(uploaded_file.name.strip(".png"),f'{LOCAL_DATA_PATH1}/test_images', target_size=(224, 224,3))
        #image_path = os.path.join(f'{LOCAL_DATA_PATH1}/test_images', f'{uploaded_file.name.strip(".png")}.png')
        with NamedTemporaryFile(dir='.', suffix='.png') as f:
            f.write(uploaded_file.getbuffer())
            #image_path = os.path.join('data/test_images', f'{uploaded_file.name.strip(".png")}.png')
            #image = load_img(image_path, target_size=(224,224,3))
            image = load_img(f.name, target_size=(224,224,3))
            image = img_to_array(image)

    submitted = st.button("Submit", key="submit_button")

    if submitted:
        st.image(uploaded_file, caption=f"{patient}'s eye image")
        #code to show processing
        with st.spinner('Classifying...'):
            #perform prediction
            prediction_result = pred(image)

        #Display prediction result
        st.write(f"Prediction: {prediction_result}")

        #stage 2 - return disease prediction if eye is unhealthy
        if prediction_result == "Unhealthy":
            with st.spinner('Classifying...'):
                prediction_result_stage2 = pred2(image)
                disease_list = ['DR', 'MH', 'DN', 'ODC', 'Others']
                disease_dict = {
                    'DR':'Diabetic retinopathy is a microvascular complication of diabetes mellitus and is a leading cause of vision loss in the elderly and working population. The image is labeled as DR if it shows any of the following clinical findings: microaneurysms, retinal dot and blot hemorrhage, hard exudates or cotton wool spots.',
                    'MH':'Media Haze: The opacity of media can be a hallmark for the presence of cataracts, vitreous opacities, corneal edema or small pupils. Moreover, some other artifacts may be introduced as a result of acquisition procedures, such as eyelash artifacts and artifacts introduced by the instrument.',
                    'DN':'Drusens are yellow or white extracellular deposits located between the retinal pigment epithelium (RPE) and Bruch’s membrane. They naturally occur in the aged population. The presence of drusen is a hallmark and early sign of significant risk of age-related macular degeneration, geographic atrophy, choroidal neovascularization, and development of RPE abnormalities.',
                    'ODC':'Optic disc cupping is the thinning of neuroretinal rim such that optic disc appears excavated. Pathological ODC is generally referred to as glaucoma. However, several other non-glaucomatous diseases, such as arteritic anterior ischemic optic neuropathy and central retinal vein occlusion. Thus, it is very important to separately evaluate ODC.',
                    'Others':'The issue is none of the following: Diabetic retinopathy, Media Haze, Drusens, Optic disc cupping. Please visit an ophthalmologist.'
                    }
                #st.write(prediction_result_stage2)
                selected_class = np.argmax(prediction_result_stage2)
                #st.write(selected_class)
                st.write(f"Anomaly: **{disease_list[selected_class]}**")
                st.write(f"\n")
                st.write(f"{disease_dict[disease_list[selected_class]]}")
