import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from gtts import gTTS
import base64

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="AgroGuard AI", page_icon="🌿", layout="centered")

# --- 2. MODEL LOADING (With Shape Mismatch Fix) ---
MODEL_PATH = "agroguard_model.h5" 

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"🚨 Model file '{MODEL_PATH}' NOT found in GitHub!")
        return None
    try:
        # Added compile=False to fix the "dense layer expects 1 input" error
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_my_model()

# --- 3. VOICE FUNCTION ---
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("diagnosis.mp3")
        with open("diagnosis.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
    except:
        pass # Silently fail if audio hits a snag

# --- 4. PREDICTION FUNCTION ---
def model_prediction(test_image):
    image = Image.open(test_image)
    image = image.resize((128, 128)) # Matches your training dimensions
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# --- 5. UI DESIGN ---
st.title("🌿 AgroGuard: Intelligent Plant Pathology")
st.markdown("---")

if model is not None:
    uploaded_file = st.file_uploader("📸 Upload a leaf image to scan...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True, caption="Uploaded Image")
        
        if st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing leaf patterns..."):
                class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                               'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                               'Tomato___healthy']
                
                result_index = model_prediction(uploaded_file)
                diagnosis = class_names[result_index].replace("___", " ").replace("_", " ")
                
                st.balloons()
                st.success(f"**Prediction:** {diagnosis}")
                speak_text(f"The AI diagnosis is {diagnosis}")
else:
    st.warning("⚠️ App is waiting for the model file to sync from GitHub.")