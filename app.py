import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 🧠 Modell laden
model = tf.keras.models.load_model("cnn_model.keras")

# 🔤 Klassen definieren (genau wie im Training)
class_names = ['Black_Rot', 'ESCA', 'Healthy', 'Leaf_Blight']

# 🎨 UI
st.set_page_config(page_title="🍇 Grape Leaf Classifier", layout="centered")
st.title("🍇 Grape Leaf Disease Classifier")
st.markdown("Lade ein Bild eines Weinblattes hoch und erhalte eine Vorhersage der Krankheitsklasse.")

# 📤 Upload
uploaded_file = st.file_uploader("Wähle ein Bild (JPG/PNG)", type=["jpg", "jpeg", "png"])

# 📏 Bild vorbereiten
img_height, img_width = 180, 180

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='📷 Hochgeladenes Bild', use_container_width=True)


    # 🖼️ Preprocessing
    img = image.resize((img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Batch dimension

    # 🔮 Vorhersage
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    # ✅ Ausgabe
    st.markdown(f"### ✅ Vorhergesagte Klasse: **{predicted_class}**")
    st.markdown(f"📊 Konfidenz: `{confidence}%`")

st.markdown("---")
st.markdown("**Made with ❤️ by Emr7y**  \nModell: CNN  \nDatenquelle: [Kaggle Dataset](https://www.kaggle.com/datasets/rm1000/augmented-grape-disease-detection-dataset)", unsafe_allow_html=True)
