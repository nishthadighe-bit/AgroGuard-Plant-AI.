import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from gtts import gTTS
import base64

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="AgroGuard AI", page_icon="🌿", layout="centered")

# --- 2. MODEL LOADING ---
MODEL_PATH = "agroguard_model.h5" 

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"🚨 Model file '{MODEL_PATH}' NOT found in GitHub!")
        return None
    try:
        # Load without local optimization metadata to reduce tensor conflicts
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Connection error: {e}")
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
            md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)
    except:
        pass 

# --- 4. PREDICTION LOGIC (The "Double-Tensor" Fix) ---
def model_prediction(test_image):
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  
    
    # FIX: We wrap the input in a list twice to satisfy the '2 input tensors' requirement
    predictions = model.predict([input_arr, input_arr]) 
    return np.argmax(predictions)

# --- 5. UI DESIGN ---
st.title("🌿 AgroGuard: Intelligent Plant Pathology")
st.markdown("---")

if model is not None:
    uploaded_file = st.file_uploader("📸 Upload a leaf image to scan...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True, caption="Uploaded Leaf")
        
        if st.button("🔍 Predict Disease"):
            with st.spinner("AI is analyzing leaf patterns..."):
                class_names = ['Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Apple Healthy', 
                               'Blueberry Healthy', 'Cherry Powdery Mildew', 'Cherry Healthy', 
                               'Corn Gray Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn Healthy', 
                               'Grape Black Rot', 'Grape Esca', 'Grape Leaf Blight', 'Grape Healthy', 
                               'Orange Greening', 'Peach Bacterial Spot', 'Peach Healthy', 
                               'Pepper Bacterial Spot', 'Pepper Healthy', 'Potato Early Blight', 
                               'Potato Late Blight', 'Potato Healthy', 'Raspberry Healthy', 'Soybean Healthy', 
                               'Squash Powdery Mildew', 'Strawberry Leaf Scorch', 'Strawberry Healthy', 
                               'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 
                               'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Spider Mites', 
                               'Tomato Target Spot', 'Tomato Yellow Leaf Curl', 'Tomato Mosaic Virus', 'Tomato Healthy']
                
                result_index = model_prediction(uploaded_file)
                diagnosis = class_names[result_index]
                
                st.balloons()
                st.success(f"**Prediction:** {diagnosis}")
                speak_text(f"The AI diagnosis is {diagnosis}")
else:
    st.info("🔄 App is preparing the AI model... Please wait.")