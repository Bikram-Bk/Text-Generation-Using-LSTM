import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Input
from tensorflow.keras.optimizers import RMSprop

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
STEP_SIZE = 3

# Prepare training data
sequences = []
next_chars = []
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sequences.append(text[i:i + SEQ_LENGTH])
    next_chars.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sequences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sequences), len(characters)), dtype=bool)
for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# Define the model
model = Sequential([
    Input(shape=(SEQ_LENGTH, len(characters))),
    LSTM(128),
    Dense(len(characters)),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# Train the model
model.fit(x, y, batch_size=128, epochs=10)

# Save the model
model.save('textgenerator_trained.keras')

# Save the optimizer state using TensorFlow's checkpointing system
checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
checkpoint.save('textgenerator_checkpoint')

print("Model and optimizer state saved.")
