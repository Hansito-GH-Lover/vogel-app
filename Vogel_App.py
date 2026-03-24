import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# STREAMLIT
# -------------------------

st.set_page_config(page_title="Vogel-Erkennung", layout="centered")
st.title("🐦 Vogel-Erkennung (stabile Version)")

# -------------------------
# MODELL LADEN
# -------------------------

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

# -------------------------
# LABELS LADEN
# -------------------------

@st.cache_resource
def load_labels():
    import json
    import urllib.request

    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    with urllib.request.urlopen(url) as f:
        return json.load(f)

labels = load_labels()

# -------------------------
# PREPROCESSING
# -------------------------

def preprocess(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# -------------------------
# UPLOAD
# -------------------------

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild")

    processed = preprocess(image)
    prediction = model.predict(processed)

    top = np.argmax(prediction)
    label = labels[str(top)][1]
    confidence = float(np.max(prediction))

    # Prüfen ob "bird" drin ist
    if "bird" in label.lower():
        st.success(f"🐦 Vogel erkannt! ({label}) - {round(confidence*100,2)}%")
    else:
        st.warning(f"Kein Vogel erkannt. ({label})")
