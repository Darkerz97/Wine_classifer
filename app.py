import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import os
import gdown
if not os.path.exists("cnn_wine_classifier.h5"):
gdown.download("https://drive.google.com/uc?id=1fPmDA3pNwIFTvGoXW0iqvsq9SAV_KDR6", "cnn_wine_classifier.h5", quiet=False)

# --- CONFIGURACIÓN DEL MODELO ---
MODEL_PATH = "cnn_wine_classifier.h5"
IMG_SIZE = (150, 150)

model = tf.keras.models.load_model(MODEL_PATH)

# --- FUNCIONES ---
def preprocess_image(img_path):
    img = Image.open(img_path).resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)[0][0]
    return "🍷 Vino" if prediction > 0.5 else "🚫 No vino"

def open_images():
    file_paths = filedialog.askopenfilenames(title="Selecciona imágenes", filetypes=[("Imágenes", "*.jpg *.png *.jpeg")])
    if not file_paths:
        return

    for widget in image_frame.winfo_children():
        widget.destroy()

    for i, path in enumerate(file_paths):
        result = predict_image(path)
        img = Image.open(path).resize((150, 150))
        tk_img = ImageTk.PhotoImage(img)

        img_label = Label(image_frame, image=tk_img)
        img_label.image = tk_img
        img_label.grid(row=i // 3, column=(i % 3)*2, padx=10, pady=10)

        result_label = Label(image_frame, text=result, font=("Arial", 12, "bold"))
        result_label.grid(row=i // 3, column=(i % 3)*2 + 1, padx=5, pady=5)

def open_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No se puede abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesar frame
        img = cv2.resize(frame, IMG_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0][0]
        label = "🍷 Vino" if prediction > 0.5 else "🚫 No vino"

        # Mostrar en pantalla
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if prediction > 0.5 else (0, 0, 255), 2)
        cv2.imshow("Clasificador de Vinos", frame)

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# --- INTERFAZ TKINTER ---
root = tk.Tk()
root.title("🍷 Clasificador de Vinos con CNN")
root.geometry("900x600")
root.config(bg="#2b2b2b")

title_label = Label(root, text="Clasificador de Vinos", font=("Arial", 20, "bold"), fg="white", bg="#2b2b2b")
title_label.pack(pady=20)

button_frame = tk.Frame(root, bg="#2b2b2b")
button_frame.pack(pady=10)

btn_load = Button(button_frame, text="📁 Cargar imágenes", command=open_images, font=("Arial", 14), bg="#4CAF50", fg="white", width=18)
btn_load.grid(row=0, column=0, padx=10)

btn_camera = Button(button_frame, text="📷 Abrir cámara", command=open_camera, font=("Arial", 14), bg="#2196F3", fg="white", width=18)
btn_camera.grid(row=0, column=1, padx=10)

# Frame para imágenes
image_frame = tk.Frame(root, bg="#2b2b2b")
image_frame.pack(fill="both", expand=True, pady=20)

root.mainloop()
