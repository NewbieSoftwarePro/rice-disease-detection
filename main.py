import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_quant (1).tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names
class_names = [
    'bacterial_leaf_blight',
    'brown_spot',
    'healthy',
    'leaf_blast',
    'leaf_scald',
    'narrow_brown_spot'
]

# Prediction function
def predict(image):
    # Preprocess the image
    img = image.convert("RGB").resize((224, 224))
    input_array = np.array(img, dtype=np.float32) / 255.0
    input_array = np.expand_dims(input_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get predicted class index and confidence
    pred_index = np.argmax(output_data)
    confidence = output_data[pred_index]

    return class_names[pred_index], confidence

# Streamlit UI
st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered")
st.title("🌾 Rice Leaf Disease Detection")

# --- Option 1: Upload from computer
uploaded_file = st.file_uploader("📂 Upload a rice leaf image", type=["jpg", "jpeg", "png"])

# --- Option 2: Capture from camera
camera_image = st.camera_input("📷 Take a picture")

# --- Select input image
image = None
if uploaded_file:
    image = Image.open(uploaded_file)
elif camera_image:
    image = Image.open(camera_image)

# --- Run prediction
if image:
    st.image(image, caption="Selected Image", use_column_width=True)

    label, confidence = predict(image)

    st.success(f"✅ Prediction: *{label}* ({confidence * 100:.2f}% confidence)")
