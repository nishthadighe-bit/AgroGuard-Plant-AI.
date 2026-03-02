import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from gtts import gTTS
import base64

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="AgroGuard AI", page_icon="🌿", layout="centered")

# --- 2. THE ULTIMATE MODEL LOADING FIX ---
MODEL_PATH = "agroguard_model.h5"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"🚨 Model file '{MODEL_PATH}' NOT found in GitHub!")
        return None
    try:
        # We use compile=False to strictly ignore the training metadata causing the '2 tensors' error
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # Manually triggering a build to ensure it's ready for 1 input
        model.build((None, 128, 128, 3)) 
        return model
    except Exception as e:
        st.error(f"❌ technical error: {e}")
        return None

model = load_my_model()

# --- 3. VOICE FEEDBACK ---
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
        pass

# --- 4. PREDICTION LOGIC ---
def model_prediction(test_image):
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  # Added normalization just in case
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# --- 5. UI DESIGN ---
st.title("🌿 AgroGuard: Intelligent Plant Pathology")
st.markdown("---")

if model is not None:
    uploaded_file = st.file_uploader("📸 Upload a leaf image to scan...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True)
        
        if st.button("🔍 Predict Disease"):
            with st.spinner("AI is analyzing..."):
                class_names = ['Apple_scab', 'Apple_black_rot', 'Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry_powdery_mildew', 'Cherry_healthy', 'Corn_gray_leaf_spot', 'Corn_common_rust', 'Corn_northern_leaf_blight', 'Corn_healthy', 'Grape_black_rot', 'Grape_esca', 'Grape_leaf_blight', 'Grape_healthy', 'Orange_greening', 'Peach_bacterial_spot', 'Peach_healthy', 'Pepper_bacterial_spot', 'Pepper_healthy', 'Potato_early_blight', 'Potato_late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_powdery_mildew', 'Strawberry_leaf_scorch', 'Strawberry_healthy', 'Tomato_bacterial_spot', 'Tomato_early_blight', 'Tomato_late_blight', 'Tomato_leaf_mold', 'Tomato_septoria_leaf_spot', 'Tomato_spider_mites', 'Tomato_target_spot', 'Tomato_yellow_leaf_curl', 'Tomato_mosaic_virus', 'Tomato_healthy']
                
                result_index = model_prediction(uploaded_file)
                diagnosis = class_names[result_index].replace("_", " ")
                
                st.balloons()
                st.success(f"**Prediction:** {diagnosis}")
                speak_text(f"The AI diagnosis is {diagnosis}")
else:
    st.warning("⚠️ App is refreshing the model connection...")