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

st.set_page_config(page_title="🍷 Clasificador de Vinos", layout="wide", page_icon="🍷")

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
    prob = float(prediction * 100)
    label = "🍷 Vino" if prediction > 0.5 else "🚫 No vino"
    return label, prob

# --- INTERFAZ STREAMLIT ---
st.title("🍷 Clasificador de Vinos con CNN")
st.markdown("Sube una imagen o usa tu cámara para clasificarla como **vino o no vino.**")

# 📁 Cargar imágenes
uploaded_files = st.file_uploader("Selecciona una o más imágenes", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("📸 Resultados de clasificación")
    cols = st.columns(3)  # tres columnas por fila

    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        label, prob = predict_image(image)
        color = "#00C851" if "Vino" in label else "#ff4444"

        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="border-radius: 10px; background-color: {color}20; padding: 15px; text-align: center; border: 2px solid {color}">
                    <img src="data:image/png;base64,{st.image(image, use_column_width=True, output_format='PNG')}" />
                    <h4 style="color:{color};">{label}</h4>
                    <p>Confianza: {prob:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")

# --- MODO CÁMARA ---
st.header("📷 Clasificación con cámara")

modo = st.radio("Selecciona el modo de cámara:", ["🌐 Modo Web (Streamlit Cloud)", "🖥️ Modo Local (solo PC)"])

if modo == "🌐 Modo Web (Streamlit Cloud)":
    camera_image = st.camera_input("Toma una foto con tu cámara")
    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="📸 Captura tomada", use_column_width=True)
        with st.spinner("🔍 Analizando imagen..."):
            label, prob = predict_image(image)
        color = "#00C851" if "Vino" in label else "#ff4444"
        st.markdown(f"<h3 style='color:{color}'>{label} — Confianza: {prob:.2f}%</h3>", unsafe_allow_html=True)

elif modo == "🖥️ Modo Local (solo PC)":
    st.info("Ejecuta esta app localmente con el comando:")
    st.code("streamlit run app.py", language="bash")

    if st.button("Abrir cámara local"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ No se pudo acceder a la cámara local.")
        else:
            st.warning("Presiona 'ESC' para cerrar la ventana de cámara.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img = cv2.resize(frame, IMG_SIZE)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                prediction = model.predict(img)[0][0]
                label = "🍷 Vino" if prediction > 0.5 else "🚫 No vino"
                color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Clasificador de Vinos", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
