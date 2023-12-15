import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from io import StringIO, BytesIO
from PIL import Image, ImageDraw, ImageOps
from ml_logic.preprocessor import load_and_preprocess_image_uploaded, load_and_preprocess_images
from ml_logic.registry import load_model
from params import *
from ml_logic.model import *
from ml_logic.registry import *
from keras import models
from keras.utils import load_img, img_to_array
from tempfile import NamedTemporaryFile


## Front page of the app displaying a form for the doctor to enter the patient names
## and upload their eye scan image. The interface then return the result i.e healthy/not Healthy.
## This function will return the downloaded image.

def pred(image_processed):
    """Display the result"""
    model = models.load_model("models/model_1.h5")
    image_processed = image_processed.reshape((1,224,224,3))
    prediction_result = model.predict(image_processed)
    #getting class index with highest probability
    # st.write(prediction_result)
    print(prediction_result)
    if prediction_result > 0.50:
        return "**Unhealthy**"
    else:
        return "**Healthy**"

#stage 2 - predict disease if eye is unhealthy
def pred2(image_processed):
    model1 = models.load_model("models/model_2.h5")
    image_processed = image_processed.reshape((1,224,224,3))
    prediction_result1 = model1.predict(image_processed)
    #getting class index with highest probability
    # st.write(prediction_result1)
    return prediction_result1

def add_rounded_edges(uploaded_file, radius=50):
    """
    Adds rounded corners to an uploaded image in Streamlit.

    Parameters:
    - uploaded_file: The file-like object uploaded via st.file_uploader.
    - radius: The radius of the rounded corners.
    """
    # Open the uploaded image file
    with Image.open(uploaded_file) as im:
        # Create a rounded mask
        mask = Image.new('L', im.size, 0)
        draw = ImageDraw.Draw(mask)
        # Draw four filled rectangles with the corner areas missing
        draw.rectangle([(0, radius), (im.width, im.height - radius)], fill=255)
        draw.rectangle([(radius, 0), (im.width - radius, im.height)], fill=255)
        # Draw four filled circles in the corner areas
        for x in (0, im.width - radius * 2):
            for y in (0, im.height - radius * 2):
                draw.pieslice([x, y, x + radius * 2, y + radius * 2], 0, 360, fill=255)
        # Apply the rounded mask to the image
        im = ImageOps.fit(im, mask.size, centering=(0.5, 0.5))
        im.putalpha(mask)

        # Save the output to a bytes object to return
        im_bytes = BytesIO()
        im.save(im_bytes, format='PNG')
        im_bytes.seek(0)

        return im_bytes

with st.container():
    st.markdown("<center><h1>Eye Spy</h1></center>", unsafe_allow_html=True)

    col_first_name, col_last_name, = st.columns(2)

    with col_first_name:
        patient_first_name = st.text_input('First Name', '', key="patient_first_name")
    with col_last_name:
        patient_last_name = st.text_input('Last Name', '', key="patient_last_name")

    uploaded_file = st.file_uploader("Select retina image file...")
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
        rounded_image = add_rounded_edges(uploaded_file)

    submitted = st.button("Submit", key="submit_button")

    if submitted:
        st.image(rounded_image, caption=f"{patient_first_name} {patient_last_name}'s retinal image.")
        #code to show processing
        with st.spinner('Classifying...'):
            #perform prediction
            prediction_result = pred(image)

        #Display prediction result
        # st.markdown(f"{prediction_result}")

        #stage 2 - return disease prediction if eye is unhealthy
        if "Unhealthy" in prediction_result:
            st.markdown("<span style='color: red; font-size: 24px;'>**Unhealthy**</span>", unsafe_allow_html=True)
            prediction_result_stage2 = pred2(image)
            # st.write(f"Anomalies: {prediction_result_stage2}")
            disease_labels = ['DR','MH','DN','ODC','Other']
            disease_list = ['Diabetic Retinopathy', 'Media Haze', 'Drusens', 'Optic Disc Cupping', 'Other']
            disease_dict = {
                'Diabetic Retinopathy':'Diabetic retinopathy is a microvascular complication of diabetes mellitus and is a leading cause of vision loss in the elderly and working population. The image is labeled as DR if it shows any of the following clinical findings: microaneurysms, retinal dot and blot hemorrhage, hard exudates or cotton wool spots.',
                'Media Haze':'Media Haze: The opacity of media can be a hallmark for the presence of cataracts, vitreous opacities, corneal edema or small pupils. Moreover, some other artifacts may be introduced as a result of acquisition procedures, such as eyelash artifacts and artifacts introduced by the instrument.',
                'Drusens':'Drusens are yellow or white extracellular deposits located between the retinal pigment epithelium (RPE) and Bruch’s membrane. They naturally occur in the aged population. The presence of drusen is a hallmark and early sign of significant risk of age-related macular degeneration, geographic atrophy, choroidal neovascularization, and development of RPE abnormalities.',
                'Optic Disc Cupping':'Optic disc cupping is the thinning of neuroretinal rim such that optic disc appears excavated. Pathological ODC is generally referred to as glaucoma. However, several other non-glaucomatous diseases, such as arteritic anterior ischemic optic neuropathy and central retinal vein occlusion. Thus, it is very important to separately evaluate ODC.',
                'Other':'The issue is none of the following: Diabetic retinopathy, Media Haze, Drusens, Optic disc cupping. Please visit an ophthalmologist.'
                }
            # st.write(prediction_result_stage2)
            selected_class = np.argmax(prediction_result_stage2)
            # st.write(selected_class)
            st.write(f"Anomaly Type: **{disease_list[selected_class]}**")
            st.write(f'Label: **{disease_labels[selected_class]}**')
            # st.write(f"\n")
            st.write(f"{disease_dict[disease_list[selected_class]]}")
        else:
            st.markdown("<span style='color: green; font-size: 24px;'>**Healthy**</span>", unsafe_allow_html=True)
