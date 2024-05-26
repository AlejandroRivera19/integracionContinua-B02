import nltk
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin


# Inicializar Flask
app = Flask(__name__)

CORS(app)

# Descargar los recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Inicializar el chatbot
chatbot = ChatBot('MiChatBot')
trainer = ChatterBotCorpusTrainer(chatbot)

# Entrenar el chatbot con el corpus en inglés y español
trainer.train('chatterbot.corpus.english')
trainer.train('chatterbot.corpus.spanish')


# Función para predecir una respuesta basada en el sentimiento del mensaje del usuario
def predecir_respuesta(mensaje_usuario):
    respuesta_chatbot = None
    if 'color' in mensaje_usuario.lower() and 'manzanas' in mensaje_usuario.lower():
        respuesta_chatbot = "Las manzanas pueden ser de varios colores, como rojo, verde o amarillo."
    return respuesta_chatbot


# Función para manejar el envío de mensajes
def enviar_mensaje(mensaje_usuario):
    respuesta_chatbot = predecir_respuesta(mensaje_usuario)
    if not respuesta_chatbot:
        respuesta_chatbot = chatbot.get_response(mensaje_usuario)
    return str(respuesta_chatbot)

# Endpoint para recibir mensajes y devolver respuestas
@app.route('/obtener_respuesta', methods=['GET'])
def obtener_respuesta():
    mensaje_usuario = request.args.get('mensaje')
    respuesta_chatbot = enviar_mensaje(mensaje_usuario)
    return jsonify({'respuesta': respuesta_chatbot})

# Ejecutar el servidor Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
