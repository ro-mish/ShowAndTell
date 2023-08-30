import io
import os
import pickle
from collections import Counter

import gdown
import numpy as np
import openai
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit_lottie
import timm
import torch
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.cluster import KMeans
from torchvision import datasets, transforms

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

openai.api_key = os.environ['OPENAI_API_KEY']

def upload_image():
    """
    Function to upload an image using Streamlit's file_uploader.
    
    Returns:
        PIL.Image or None: A PIL Image if an image is uploaded, otherwise None.
    """
    uploaded_file = st.file_uploader("Choose an image of your favorite portrait!", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        #st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Image successfully uploaded!")
        st.image(image)
        return image
    
    return None

def load_image(img, transform=None):
    """ 
    Function to load an image from a file path and transform it.
    
    Args:
        img_path (str): Path to the image file.
        transform (torchvision.transforms): Transform to apply to the image.
        
    Returns:    
        torch.Tensor: A PyTorch tensor of the image.
    """
    # Load the image
    if transform is not None:
        img = transform(img)
    
    img = img.unsqueeze(0)
    
    return img

@st.cache_resource
def load_model(path):
    """ 
    Function to load a PyTorch model from a file path.
    
    Args:
        path (str): Path to the model file.
    
    Returns:    
        torch.nn.Module: A PyTorch model.
    """
    new_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).to("cpu")
    
    url = "https://drive.google.com/uc?export=download&id=19TsWdOq6a2lRkLoBuaJGw_8zRhL7cGox"
    
    buffer = io.BytesIO()
    
    gdown.download(url, output=buffer, quiet=False)
    
    buffer.seek(0)
    
    new_model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
    
    new_model.eval()
    
    return new_model


def get_top_colors(img, factor=1.1, num=5):
    """
    Uses Looks at the top occurring colors in the image.
    
    Args:
        img_path (str): Path to the image file.
        
    Returns:                
        list: A list of the top `num` colors in the image.
    """

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
    Returns an image with concatenated squares for each color input.
    
    Args:   
        top_colors (list): A list of RGB tuples for the top colors in the image.
        
    Returns:        
        PIL.Image: An image with the top colors displayed in squares.
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

def get_unique_colors(img, k=6):
    """
    Use k-means clustering to find the k most "unique" colors in the image.
    
    Args:
        img (PIL.Image): Image to find the unique colors of.
        k (int): Number of unique colors to find.

    Returns:
        list: A list of RGB values.
    """
    
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
    Creates an image with the given colors.
    
    """
    color_count = len(colors)
    img = Image.new('RGB', (color_count * 100, 100))
    draw = ImageDraw.Draw(img)
    
    for idx, color in enumerate(colors):
        draw.rectangle([idx * 100, 0, (idx + 1) * 100, 100], fill=tuple(color))
    
    return img

def write_screenplay(emotion:str, colors:list, director:str="Christopher Nolan"):
    output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": f"You are a successful set-designer that has won multiple awards. You have been given a reference image, that reveals a character's emotion as: {emotion}\
            You are currently working on a new project."},
          {"role": "user", "content": f"Write a highly descriptive set design that will fit the style of {director}. \
            You are given a reference image of the character. You are not to assume the character's gender. The character is visibly {emotion}.\
            based on the following colors in RGB format: {str(colors)}. You are to write descriptive set design for the character's room, and reference colors by name only."},
      ]
    )
    return output


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
    st.lottie("https://lottie.host/86a4be01-7274-4cdc-878c-1040815eb450/sowTE5AlrP.json", height=10, width=10)
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
    




