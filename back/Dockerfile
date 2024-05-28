# Utiliza la imagen base de Python
FROM python:3.8.12

# Instala las bibliotecas necesarias para el complemento de plataforma "xcb"
RUN apt-get update && apt-get install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx python3-tk libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Instala las bibliotecas necesarias para NLTK y ChatterBot
RUN pip install --upgrade pip
RUN pip install --upgrade nltk
RUN pip install nltk
RUN pip install --use-pep517 chatterbot
RUN pip install --use-pep517 chatterbot-corpus
RUN pip install flask
RUN pip install flask-cors
RUN pip install wikipedia-api
# Descargar los recursos necesarios de NLTK
RUN python -m nltk.downloader vader_lexicon

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY chatbotV5.py /app/

# Ejecuta el script cuando se inicie el contenedor
CMD ["python", "chatbotV5.py"]