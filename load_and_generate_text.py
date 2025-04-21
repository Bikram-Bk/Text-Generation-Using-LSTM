import os
import warnings
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Suppress specific warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='keras.layers.rnn.rnn')

# Download and preprocess text data
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]
characters = sorted(set(text))
char_to_index = {char: i for i, char in enumerate(characters)}
index_to_char = {i: char for i, char in enumerate(characters)}

SEQ_LENGTH = 40

# Function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function to generate text
def generate_text(model, length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(characters)), dtype=bool)
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_pred, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

# Load the model
print("Loading the model...")
model = load_model('textgenerator_trained.keras')
print("Model loaded.")

# Load the optimizer state using TensorFlow's checkpointing system
print("Loading optimizer state...")
checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
checkpoint.restore(tf.train.latest_checkpoint('.')).expect_partial()
print("Optimizer state loaded.")

# Generate text with different temperatures
print('---------0.2---------')
print(generate_text(model, 300, 0.2))
print('---------0.4---------')
print(generate_text(model, 300, 0.4))
print('---------0.6---------')
print(generate_text(model, 300, 0.6))
print('---------0.8---------')
print(generate_text(model, 300, 0.8))
print('---------1.0---------')
print(generate_text(model, 300, 1.0))
