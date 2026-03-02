import streamlit as st
import tensorflow as tf
from gtts import gTTS
import numpy as np
from PIL import Image
import base64
import os

# 1. Page Configuration for a Professional Look
st.set_page_config(page_title="AgroGuard AI", page_icon="🌿", layout="centered")

# Custom CSS for a better "Vibe"
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 AgroGuard: Intelligent Plant Pathology")
st.markdown("### Multi-Modal Disease Detection System")

# 2. Load the Model Safely
@st.cache_resource
def load_my_model():
    # Use compile=False to avoid version mismatch errors during deployment
    return tf.keras.models.load_model('agroguard_model.h5', compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"⚠️ Model file 'agroguard_model.h5' not found in your GitHub repo!")

# 3. Comprehensive Labels (Matching your dataset)
labels = [
    'Apple: Scab', 'Apple: Black rot', 'Apple: Cedar apple rust', 'Apple: Healthy', 
    'Blueberry: Healthy', 'Cherry: Powdery mildew', 'Cherry: Healthy', 
    'Corn: Gray leaf spot', 'Corn: Common rust', 'Corn: Northern Leaf Blight', 'Corn: Healthy', 
    'Grape: Black rot', 'Grape: Esca (Black Measles)', 'Grape: Leaf blight', 'Grape: Healthy', 
    'Orange: Haunglongbing (Citrus greening)', 'Peach: Bacterial spot', 'Peach: Healthy', 
    'Pepper: Bacterial spot', 'Pepper: Healthy', 'Potato: Early blight', 'Potato: Late blight', 
    'Potato: Healthy', 'Raspberry: Healthy', 'Soybean: Healthy', 'Squash: Powdery mildew', 
    'Strawberry: Leaf scorch', 'Strawberry: Healthy', 'Tomato: Bacterial spot', 
    'Tomato: Early blight', 'Tomato: Late blight', 'Tomato: Leaf Mold', 
    'Tomato: Septoria leaf spot', 'Tomato: Spider mites', 'Tomato: Target Spot', 
    'Tomato: Yellow Leaf Curl Virus', 'Tomato: Mosaic virus', 'Tomato: Healthy'
]

# 4. Voice Feedback Function
def play_voice(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("temp.mp3")
        with open("temp.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)
        os.remove("temp.mp3")
    except Exception as e:
        st.write("🔈 Audio feedback unavailable.")

# 5. User Interface for Uploading
uploaded_file = st.file_uploader("📸 Upload a leaf image to scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Scanning leaf...', use_container_width=True)
    
    # PREPROCESSING: Resize to 128x128 to match your model's training
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # PREDICTION
    with st.spinner('Analyzing plant tissue...'):
        predictions = model.predict(img_array)
        result_index = np.argmax(predictions)
        
        # Guardrail: Check if prediction index is within labels range
        if result_index < len(labels):
            disease_name = labels[result_index]
            confidence = np.max(predictions) * 100
            
            # RESULTS DISPLAY
            st.balloons() # Celebration effect!
            st.success(f"**Diagnosis:** {disease_name}")
            st.info(f"**Confidence Level:** {confidence:.2f}%")
            
            # VOICE TRIGGER
            play_voice(f"Diagnosis complete. Detected {disease_name}")
        else:
            st.warning("Model predicted a category outside the current label list.")