import io
import os
import pickle
from collections import Counter

import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit_lottie
import timm
import torch
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.cluster import KMeans
from torchvision import datasets, transforms

from utils import *

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

openai.api_key = os.environ['OPENAI_API_KEY']

id_2_label = {
        0: 'Angry',
        1: 'Disgusted',
        2: 'Fearful',
        3: 'Happy',
        4: 'Neutral',
        5: 'Sad',
        6: 'Surprised'
    }


col1, col2 = st.columns([2,1], gap="large")

im_check = False
    
if __name__ == "__main__":

    st.title("Show & Tell")
    
    st.caption("A tool to generate a set design based on the emotion & colors present in an image of person. \
        Get started by uploading a \
        Your data is never persisted anywhere.")
    
    st.header("Upload an image to get started")
    image = upload_image()
    #st.lottie("https://lottie.host/86a4be01-7274-4cdc-878c-1040815eb450/sowTE5AlrP.json")
    #st.lottie("https://lottie.host/a33a008b-43d1-478b-ae0f-5e81a9fef8ff/7QxIUNpemB.json")
    #st.lottie("https://lottie.host/643212bb-8df7-4d3e-8e87-54d156fbde5e/ND0hDloRtz.json")
    
    # Define the same transformation as during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    PATH='model.pkl'
    with st.spinner("Getting the model from the pit of the Mariana Trench..."):
        model = load_model(PATH)
    
    if image:
        im_check = True
        
        tc = get_unique_colors(image)
        
        image = load_image(image, transform=transform)
        with st.spinner("Getting the model from the pit of the Mariana Trench..."):
        
            outputs = model(image)
            outputs = outputs.detach().numpy().flatten()
            
        st.write(f"Predicted emotion from image: {id_2_label[np.argmax(outputs)]}")
        options = ["Christopher Nolan", "Quentin Tarantino", "Steven Spielberg", "Martin Scorsese"]
        selected_option = None
        
        st.header("Select a director to generate a set design")
        selected_option = st.selectbox("Select a director", options)

        with st.expander("Show model details and colors", expanded=False):
            fig = px.bar(x=list(id_2_label.values()), y=outputs, labels={'x':'Emotion Category', 'y': 'Activation'})
            st.plotly_chart(fig)
            
            st.write("Top colors in the image:")
            st.image(visualize_colors(tc))     
        
        
        with st.expander("Generated set description", expanded=False):
            with st.spinner("Taking this info to the writer's room..."):
                st.write(write_screenplay(emotion=id_2_label[np.argmax(outputs)], colors=tc, director=selected_option)["choices"][0]["message"]["content"])
    

           