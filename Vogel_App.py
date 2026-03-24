import streamlit as st
from ultralytics import YOLO
from PIL import Image

# -------------------------
# STREAMLIT SETTINGS
# -------------------------

st.set_page_config(page_title="Vogel-Erkennung", layout="centered")
st.title("🐦 Vogel-Erkennung mit KI")

st.write("Lade ein Bild hoch und die KI sagt dir, ob ein Vogel erkannt wurde.")

# -------------------------
# MODELL LADEN
# -------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # kleines Modell

model = load_model()

# -------------------------
# UPLOAD
# -------------------------

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# ANALYSE
# -------------------------

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Dein Bild", use_column_width=True)

    results = model(image)

    found_bird = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])

        if label == "bird":
            found_bird = True
            st.success(f"🐦 Vogel erkannt! Sicherheit: {round(confidence*100, 2)} %")

    if not found_bird:
        st.warning("❌ Kein Vogel erkannt.")
