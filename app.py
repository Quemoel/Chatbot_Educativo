from flask import Flask, render_template, request, jsonify
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Carregar os dados e preparar o modelo
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

patterns = []
responses = []
tags = []
for intent in intents['intents']:
    for pattern in intent['padroes']:
        patterns.append(pattern)
        responses.append(intent['respostas'])
        tags.append(intent['tag'])

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)
sequences_padded = pad_sequences(sequences, padding='post', maxlen=20)  # Fix the maxlen to a value suitable for your data

responses_onehot = np.zeros((len(tags), len(tags)), dtype='int32')
for i, tag in enumerate(tags):
    responses_onehot[i][i] = 1

model = keras.Sequential([
    keras.layers.Embedding(len(word_index) + 1, 128),
    keras.layers.LSTM(256, return_sequences=True),
    keras.layers.LSTM(256),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(tags), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(sequences_padded, responses_onehot, epochs=150, batch_size=8)

def preprocess_input(user_input):
    # Implement any necessary preprocessing for the user input here
    # For example, convert the input to lowercase and remove special characters
    return user_input.lower()

def chatbot_response(user_input):
    user_input = preprocess_input(user_input)
    sequence = tokenizer.texts_to_sequences([user_input])
    sequence_padded = pad_sequences(sequence, padding='post', maxlen=len(sequences_padded[0]))
    predicted_index = np.argmax(model.predict(sequence_padded), axis=-1)[0]
    tag = tags[predicted_index]
    responses_list = responses[predicted_index]
    return random.choice(responses_list)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
