import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------------------
# Load TFLite Model D:\Rice_leaves_detection\
# ---------------------------
interpreter = tf.lite.Interpreter(model_path="model_quant (1).tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class Names
class_names = [
    'bacterial_leaf_blight',
    'brown_spot',
    'healthy',
    'leaf_blast',
    'leaf_scald',
    'narrow_brown_spot'
]

# ---------------------------
# Prediction Function
# ---------------------------
def predict(image):
    img = image.convert("RGB").resize((224, 224))
    input_array = np.array(img, dtype=np.float32) / 255.0
    input_array = np.expand_dims(input_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    pred_index = np.argmax(output_data)
    return class_names[pred_index]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🌾 Rice Leaf Disease Detector", layout="centered")

# Title and Description
st.title("🌾 Rice Leaf Disease Detector")
st.write("Upload a rice leaf image or take a picture to detect the disease using a **MobileNetV2 model**.")

# Sidebar - About Section
st.sidebar.title("ℹ️ About")
st.sidebar.write("""
This app helps detect **common rice leaf diseases** using AI.  
It is powered by a **MobileNetV2 model** converted to TensorFlow Lite.  
""")


# File Upload or Camera
uploaded_file = st.file_uploader("📂 Upload a rice leaf image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Or take a picture")

# Select Input Image
image = None
if uploaded_file:
    image = Image.open(uploaded_file)
elif camera_image:
    image = Image.open(camera_image)

# Show Image Preview
if image:
    st.image(image, caption="🌱 Selected Leaf", use_container_width=True)

# Instructions
with st.expander("📌 How to use this app?"):
    st.write("""
    1. Upload a rice leaf image **or take a picture**.  
    2. Click **Predict** to analyze the leaf.  
    """)

# Prediction Button
if st.button("🔍 Predict"):
    if image is not None:
        with st.spinner("Analyzing image... Please wait ⏳"):
            label = predict(image)

        # Styled Results
        if label.lower() == "healthy":
            st.success("🌱 The leaf is **Healthy** ✅")
        else:
            st.error(f"⚠️ Disease Detected: **{label}**")
    else:
        st.warning("⚠️ Please upload or capture an image first!")


