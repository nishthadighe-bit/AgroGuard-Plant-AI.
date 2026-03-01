import streamlit as st
import tensorflow as tf
from gtts import gTTS
import numpy as np
from PIL import Image
import base64
import os

# 1. Page Configuration
st.set_page_config(page_title="AgroGuard AI", page_icon="🌿")
st.title("🌿 AgroGuard: Intelligent Plant Pathology")
st.markdown("### Multi-Modal Disease Detection System")

# 2. Load the Model (With the Version Fix)
@st.cache_resource
def load_my_model():
    # Adding compile=False fixes the 'dense layer' error
    return tf.keras.models.load_model('agroguard_model.h5', compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Labels
labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

def play_voice(text):
    tts = gTTS(text=text, lang='en')
    tts.save("temp.mp3")
    with open("temp.mp3", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_allow_html=True)
    os.remove("temp.mp3")

uploaded_file = st.file_uploader("Upload a leaf image to scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Scanning leaf...', use_container_width=True)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    result_index = np.argmax(predictions)
    disease_name = labels[result_index].replace('___', ': ').replace('_', ' ')
    confidence = np.max(predictions) * 100
    
    st.success(f"**Diagnosis:** {disease_name}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    play_voice(f"Detected {disease_name}")
