import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from groq import Groq
from dotenv import load_dotenv # Para cargar variables de entorno desde un archivo .env

# --- Configuración global --- 
IMG_HEIGHT = 180
IMG_WIDTH = 180
class_names = ['amarillento', 'phoma', 'quemaduras', 'rut', 'sanas'] # ¡Importante! Coincide con tu entrenamiento
MODEL_PATH = 'image_classifier_model.keras'

# Cargar variables de entorno (para GROQ_API_KEY fuera de Colab)
load_dotenv()

# --- Cargar el modelo entrenado ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de que el archivo 'image_classifier_model.keras' esté en el mismo directorio.")
    sys.exit(1)

# --- Función para predecir y analizar con IA ---
def analizar_imagen_con_ia(image_path):
    # 1. Configurar Cliente Groq
    try:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("La variable de entorno 'GROQ_API_KEY' no está configurada.")
        client_ai = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Error al configurar la API de Groq: {e}\nAsegúrate de que 'GROQ_API_KEY' esté en tus variables de entorno o en un archivo .env."

    # 2. Procesar imagen y predecir con el modelo
    if not os.path.exists(image_path):
        return f"Error: La imagen '{image_path}' no se encontró."

    try:
        img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Crear un lote (batch)

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
            model="llama-3.1-8b-instant", # O el modelo de Groq que prefieras
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
