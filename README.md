Detección de Enfermedades en Plantas con Visión Artificial e IA
Este proyecto implementa un sistema para la detección de enfermedades en hojas de plantas utilizando un modelo de clasificación de imágenes basado en TensorFlow/Keras y un análisis experto potenciado por la API de Groq.

📋 Contenido
Setup e Instalación: Dependencias y configuración de la API de Groq.
Preparación del Dataset: Carga, extracción y preprocesamiento de imágenes.
Creación y Entrenamiento del Modelo: Definición y entrenamiento de la CNN.
Evaluación del Modelo: Análisis de la precisión y pérdida del modelo.
Análisis Experto con IA (Groq): Integración del modelo con Groq para diagnósticos y recomendaciones.
Despliegue Local (app.py): Cómo ejecutar la aplicación fuera de Colab.
1. Setup e Instalación
Para empezar, instalamos las librerías y herramientas necesarias:

Extracción de archivos .rar: Utilizamos apt-get install -y unrar para poder descomprimir el dataset de imágenes.
Librerías Python: Instalamos groq tensorflow numpy Pillow python-dotenv matplotlib usando pip. Estas librerías son para:
groq: Interactuar con la API de Groq para el análisis experto.
tensorflow: Construir y entrenar el modelo de visión artificial.
numpy: Manejo eficiente de arrays numéricos.
Pillow: Procesamiento y manipulación de imágenes.
python-dotenv: Para cargar variables de entorno (como la API key de Groq) en despliegues locales.
matplotlib: Para visualizar los resultados (ej. gráficas de entrenamiento, imágenes de ejemplo).
También definimos un archivo requirements.txt con estas dependencias, útil para replicar el entorno de forma sencilla:

tensorflow==2.16.1
numpy==1.25.2
Pillow==10.3.0
groq==0.8.0
matplotlib==3.7.1
python-dotenv==1.0.1
Configuración de la API de Groq
La clave de la API de Groq (GROQ_API_KEY) se almacena de forma segura en los Secretos de Google Colab para evitar que se exponga en el código. El notebook verifica que esta clave esté configurada correctamente para poder usar la IA generativa de Groq.

2. Preparación del Dataset
Carga y Extracción: Se sube un archivo .rar (que contiene el dataset de imágenes de hojas) a Colab y se descomprime.
Conversión de Imágenes: Aunque en este caso el dataset ya contenía imágenes PNG, el código incluye una celda para convertir imágenes .webp a .png si fuera necesario, usando la librería Pillow.
Carga con TensorFlow: Las imágenes se cargan en un objeto tf.data.Dataset usando tf.keras.utils.image_dataset_from_directory. Se definen parámetros como el tamaño de las imágenes (180x180 píxeles), el modo de etiquetas (categorical para one-hot encoding) y el tamaño del batch inicial (1 para facilitar la división).
Clases: Se extraen los nombres de las clases (tipos de enfermedades o 'sanas') del dataset.
Visualización: Se muestran algunas imágenes de muestra del dataset con sus etiquetas correspondientes para verificar que la carga fue exitosa.
División de Datos: El dataset se divide en conjuntos de entrenamiento (80%), validación (10%) y prueba (10%). Esto es crucial para entrenar el modelo, ajustarlo y evaluarlo de forma imparcial.
Optimización del Pipeline: Los datasets se configuran para un mejor rendimiento usando cache() y prefetch() de TensorFlow, lo que acelera el proceso de entrenamiento al mantener los datos listos.
3. Creación y Entrenamiento del Modelo
Modelo CNN: Se construye un modelo de Red Neuronal Convolucional (CNN) usando tf.keras.Sequential.
Aumento de Datos: Incluye capas como RandomFlip, RandomRotation y RandomZoom para generar variaciones de las imágenes de entrenamiento y hacer el modelo más robusto.
Reescalado: Una capa Rescaling normaliza los valores de los píxeles al rango [0, 1].
Capas Convolucionales y MaxPooling: Múltiples bloques de Conv2D y MaxPooling2D extraen características de las imágenes.
Capas Densa: Una capa Flatten convierte las características en un vector, seguido de capas Dense para la clasificación final, con una capa de salida softmax para predecir probabilidades entre las clases.
Compilación: El modelo se compila usando el optimizador adam, la función de pérdida CategoricalCrossentropy (adecuada para etiquetas one-hot) y se monitoriza la accuracy.
Entrenamiento: El modelo se entrena durante 15 EPOCHS (épocas) utilizando los conjuntos de entrenamiento y validación.
Guardado del Modelo: Una vez entrenado, el modelo se guarda en el formato .keras para poder ser reutilizado sin necesidad de volver a entrenarlo.
4. Evaluación del Modelo
Se visualiza el historial de entrenamiento mediante gráficos de matplotlib. Esto nos permite analizar cómo la accuracy (precisión) y la loss (pérdida) evolucionaron en los conjuntos de entrenamiento y validación a lo largo de las épocas, ayudando a detectar sobreajuste o subajuste.

5. Análisis Experto con IA (Groq)
Se desarrolla una función analizar_enfermedad_con_ia() que combina la capacidad del modelo de visión con la inteligencia generativa de Groq:

Selección de Imagen: Elige una imagen aleatoria del dataset.
Predicción del Modelo: El modelo de visión predice la clase de la enfermedad y la confianza.
Consulta a Groq: Se construye un prompt para Groq, pidiéndole que actúe como un experto fitopatólogo. Si detecta una enfermedad, Groq explica sus síntomas y ofrece 3 consejos de tratamiento. Si la planta está 'sana', Groq felicita al agricultor.
Visualización y Texto: Muestra la imagen, la predicción del modelo y la respuesta detallada del experto IA de Groq.
6. Despliegue Local (app.py)
Finalmente, se proporciona un script app.py que permite ejecutar la lógica de detección de enfermedades y análisis de Groq en un entorno local. Este script:

Carga el modelo image_classifier_model.keras.
Utiliza python-dotenv para cargar la GROQ_API_KEY desde un archivo .env.
Define una función para analizar una imagen dada, predecirla con el modelo, y obtener recomendaciones de Groq.
Permite ejecutarlo desde la línea de comandos pasando la ruta de una imagen como argumento.
Estructura de archivos para el despliegue local:

mi_aplicacion_plantas/
├── app.py
├── requirements.txt
├── image_classifier_model.keras  # Tu modelo entrenado
└── .env                        # Archivo para la GROQ_API_KEY
Contenido de .env:

GROQ_API_KEY=tu_clave_api_de_groq_aqui
Instrucciones para ejecutar app.py localmente:

Guarda los archivos image_classifier_model.keras, app.py, requirements.txt y crea el archivo .env en el mismo directorio.
Instala las dependencias Python (pip install -r requirements.txt).
Ejecuta desde la terminal: python app.py "./ruta/a/tu/imagen.png".
