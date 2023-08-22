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


def upload_image():
    """
    Function to upload an image using Streamlit's file_uploader.
    
    Returns:
        PIL.Image or None: A PIL Image if an image is uploaded, otherwise None.
    """
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        #st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Image successfully uploaded!")
        return image
    
    return None

def load_image(img, transform=None):
    #img = Image.open(image_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    st.image(image)
    # Add the batch dimension
    img = img.unsqueeze(0)
    return img

@st.cache_resource
def load_model(path):
    # new_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).to("cpu")
    # # Load the weights
    # new_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # new_model.eval()  # Set the model to evaluation mode
    url = "https://drive.google.com/uc?export=download&id=1Ia58B9ynHYYIQByeB5lSoqZi6RumjA6T"
    # output = 'emotion_model.pkl'
    # gdown.download(url, output, quiet=False)
    
    # with open('emotion_model.pkl', 'rb') as f:
    #     new_model = pickle.load(f)
    
    #file_url = 'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID'  # Replace with your actual link

    # Create an in-memory binary stream
    buffer = io.BytesIO()

    # Download file into the buffer
    gdown.download(url, output=buffer, quiet=False)

    # Reset buffer position to the beginning
    buffer.seek(0)

    # Load the pickled data from the buffer
    new_model = pickle.load(buffer)
    
    return new_model


def get_top_colors(img, factor=1.1, num=5):
    """
    Returns the top `num` colors in the image at `img_path`.
    """
    # # Load image and convert to RGB
    # img = Image.open(img_path).convert('RGB')
    img = img.convert('RGB')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    # Convert image data to list of tuples
    data = list(img.getdata())
    
    # Count occurrences of each color
    color_count = Counter(data)
    
    # Get the most common colors
    top_colors = [item[0] for item in color_count.most_common(num)]
    st.write(top_colors)
    return top_colors

def create_color_squares(top_colors, square_size=100):
    """
    Returns an image with concatenated squares for each color in `top_colors`.
    """
    width = square_size * len(top_colors)
    height = square_size
    
    # Create a new blank image with the computed width and height
    img = Image.new('RGB', (width, height))
    
    draw = ImageDraw.Draw(img)
    
    for idx, color in enumerate(top_colors):
        draw.rectangle(
            [idx * square_size, 0, (idx + 1) * square_size, height],
            fill=color
        )
    
    return img

def get_unique_colors(img, k=5):
    """
    Use k-means clustering to find the k most "unique" colors in the image.
    """
    # Load the image
    
    
    # Convert the image to an array of RGB values
    img_data = np.array(img)
    pixels = img_data.reshape(-1, 3)
    
    # Use k-means clustering to find k unique colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Extract the cluster centers
    unique_colors = kmeans.cluster_centers_
    
    # Convert to integers
    unique_colors = unique_colors.round(0).astype(int)
    
    return unique_colors.tolist()

def visualize_colors(colors):
    """
    Create an image displaying the unique colors side by side.
    """
    color_count = len(colors)
    img = Image.new('RGB', (color_count * 100, 100))
    draw = ImageDraw.Draw(img)
    
    for idx, color in enumerate(colors):
        draw.rectangle([idx * 100, 0, (idx + 1) * 100, 100], fill=tuple(color))
    
    return img


id_2_label = {
        0: 'Angry',
        1: 'Disgusted',
        2: 'Fearful',
        3: 'Happy',
        4: 'Neutral',
        5: 'Sad',
        6: 'Surprised'
    }

PATH='model.pkl'
col1, col2 = st.columns(2)
tab2, tab1 = st.tabs(["Model Results", "Top colors"])

if __name__ == "__main__":
    
    model = load_model(PATH)
    st.title("Show and Tell")
    with st.sidebar:
        
        st.header("Upload a picture of a face to predict the person's emotion!")
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

    st.subheader("Predicted emotion")
    
    if image:
        
        tc = get_unique_colors(image)
        
        image = load_image(image, transform=transform)
        
        outputs = model(image)
        
        st.subheader(f"Predicted emotion from image: {id_2_label[torch.argmax(outputs).item()]}")
        

        with tab1:
            st.write("Top colors in the image:")
            st.image(visualize_colors(tc))
              
        with tab2:
            fig = px.bar(x=list(id_2_label.values()), y=outputs.detach().numpy().flatten(), labels={'x':'Emotion Category', 'y': 'Probability'})
            st.plotly_chart(fig)
        
        
            

    