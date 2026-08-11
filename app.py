import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from groq import Groq
import streamlit as st
import gdown

# --- Configuración de la página ---
st.set_page_config(page_title="Detección de Enfermedades", page_icon="🌿")

# --- Configuración global --- 
IMG_HEIGHT = 180
IMG_WIDTH = 180
class_names = ['amarillento', 'phoma', 'quemaduras', 'rut', 'sanas']
MODEL_PATH = 'image_classifier_model.keras'
GDRIVE_FILE_ID = '1aRrqWp4H_a6qRmWDLt65DKHysPhKUS00'

# --- Cargar el modelo en memoria ---
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo desde Google Drive... Esto solo ocurre una vez."):
            temp_file = "temp_model.keras"
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, temp_file, quiet=False)
            os.rename(temp_file, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# --- Interfaz de usuario en Streamlit ---
st.title("🌿 Detección de Enfermedades en Hojas")
st.write("Suba una imagen para clasificar el estado de la hoja y obtener un diagnóstico con IA.")

# Componente para subir archivo desde el navegador
uploaded_file = st.file_uploader("Selecciona una imagen de hoja...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Guardar archivo temporal para procesarlo con TensorFlow
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Mostrar la imagen cargada
    st.image(uploaded_file, caption="Imagen seleccionada", use_container_width=True)

    if st.button("🔍 Analizar Enfermedad"):
        with st.spinner("Clasificando imagen y consultando al experto IA..."):
            try:
                # 1. Predicción con el modelo
                img = image.load_img(tmp_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                predicted_class = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                st.success(f"**Resultado:** {predicted_class.upper()} ({confidence:.1f}% de confianza)")

                # 2. Diagnóstico con Groq
                api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
                if not api_key:
                    st.error("❌ La clave GROQ_API_KEY no está configurada en los Secrets de Streamlit.")
                else:
                    client_ai = Groq(api_key=api_key)
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

                    st.markdown("### 🌿 Análisis del Experto IA")
                    st.info(respuesta_ia)

            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
