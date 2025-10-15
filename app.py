import os
import numpy as np
import tensorflow as tf
import gdown
import streamlit as st
from PIL import Image
import cv2

# --- CONFIGURACIÓN GLOBAL ---
MODEL_PATH = "cnn_wine_classifier.h5"
IMG_SIZE = (150, 150)
MODEL_URL = "https://drive.google.com/uc?id=1WyuBovJBWX5SM8BAoIz9sNxoYaoq6B2f"

# --- DESCARGA AUTOMÁTICA DEL MODELO ---
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Modelo no encontrado, descargando desde Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("✅ Modelo descargado correctamente.")
    except Exception as e:
        st.error("❌ Error al descargar el modelo. Verifica permisos o conexión.")
        st.stop()

# --- CARGA DEL MODELO ---
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.sidebar.success("✅ Modelo cargado correctamente.")
except Exception as e:
    st.sidebar.error("❌ Error al cargar el modelo. Verifica el archivo .h5 o su versión.")
    st.sidebar.text(e)
    st.stop()

# --- FUNCIONES ---
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img):
    processed = preprocess_image(img)
    prediction = model.predict(processed)[0][0]
    return "🍷 Vino" if prediction > 0.5 else "🚫 No vino"

# --- INTERFAZ STREAMLIT ---
st.title("🍷 Clasificador de Vinos con CNN")
st.markdown("Sube una imagen de una botella de vino para clasificarla como **vino o no vino.**")

# 📁 Cargar imágenes
uploaded_files = st.file_uploader("Selecciona una o más imágenes", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"🖼️ {uploaded_file.name}", use_column_width=True)

        with st.spinner("Analizando imagen..."):
            result = predict_image(image)

        st.subheader(f"Resultado: {result}")

# 📷 Cámara (opcional — no funciona en Streamlit Cloud)
st.markdown("---")
st.subheader("📷 Clasificación por cámara (solo local)")
st.info("Esta función solo funciona si ejecutas la app localmente (`streamlit run app.py`).")

if st.button("Abrir cámara"):
    st.warning("⚠️ Esta opción no funciona en Streamlit Cloud. Ejecuta localmente para usarla.")
    # Si quisieras habilitarla localmente, descomenta esto:
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     st.error("No se puede abrir la cámara")
    # else:
    #     st.info("Presiona 'ESC' para salir de la cámara.")
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         img = cv2.resize(frame, IMG_SIZE)
    #         img = img / 255.0
    #         img = np.expand_dims(img, axis=0)
    #         prediction = model.predict(img)[0][0]
    #         label = "🍷 Vino" if prediction > 0.5 else "🚫 No vino"
    #         color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
    #         cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    #         cv2.imshow("Clasificador de Vinos", frame)
    #         if cv2.waitKey(1) & 0xFF == 27:
    #             break
    #     cap.release()
    #     cv2.destroyAllWindows()
