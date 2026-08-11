import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from groq import Groq
import streamlit as st
import gdown

# --- Configuración global --- 
IMG_HEIGHT = 180
IMG_WIDTH = 180
class_names = ['amarillento', 'phoma', 'quemaduras', 'rut', 'sanas'] # Coincide con tu entrenamiento
MODEL_PATH = 'image_classifier_model.keras'

# ID de tu archivo subido a Google Drive
GDRIVE_FILE_ID = '1aRrqWp4H_a6qRmWDLt65DKHysPhKUS00'

# --- Función optimizada para descargar y cargar el modelo ---
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando el modelo desde Google Drive... Esto solo se hace una vez."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"❌ Error al cargar/descargar el modelo: {e}")
    st.stop()

# --- Función para predecir y analizar con IA ---
def analizar_imagen_con_ia(image_path):
    # 1. Configurar Cliente Groq (Lee desde st.secrets de Streamlit o variables de entorno)
    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not api_key:
            raise ValueError("La clave 'GROQ_API_KEY' no está configurada.")
        client_ai = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Error al configurar la API de Groq: {e}\nAsegúrate de configurar 'GROQ_API_KEY' en los Secrets de Streamlit Cloud."

    # 2. Procesar imagen y predecir con el modelo
    if not os.path.exists(image_path):
        return f"Error: La imagen '{image_path}' no se encontró."

    try:
        img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Crear lote (batch)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
    except Exception as e:
        return f"Error al procesar la imagen o realizar la predicción: {e}"

    # 3. Consultar a Groq (IA) sobre la enfermedad detectada
    try:
        prompt = f"""Actúa como un experto fitopatólogo.
        El modelo de visión ha detectado la clase '{predicted_class}' con un {confidence:.2f}% de confianza.
        Si la clase es 'sanas', felicita al agricultor.
        Si es una enfermedad, explica qué es, sus síntomas y da 3 consejos breves de tratamiento. Sé conciso y claro."""

        chat_completion = client_ai.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        respuesta_ia = chat_completion.choices[0].message.content
    except Exception as e:
        respuesta_ia = f"Error al obtener respuesta de Groq: {e}"

    # 4. Construir el resultado
    result = f"Detección para la imagen '{image_path}':\n"
    result += f"Clase predicha: {predicted_class} con {confidence:.1f}% de confianza.\n\n"
    result += f"--- 🌿 ANÁLISIS DEL EXPERTO IA ---\n"
    result += respuesta_ia

    return result

# --- Ejecución principal del script ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python app.py <ruta_a_la_imagen>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    print(analizar_imagen_con_ia(input_image_path))
