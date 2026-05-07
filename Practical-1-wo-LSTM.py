import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        print("  ", filename)

import pandas as pd

df = pd.read_csv("/kaggle/input/imdb-dataset/IMDB Dataset.csv")

print(df.head())
print(df.shape)

df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

texts = df['review'].values
labels = df['sentiment'].values

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding='post'
)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    padded_sequences,
    labels,
    test_size=0.2,
    random_state=42
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

sample_review = ["The movie was boring and a waste of time"]

seq = tokenizer.texts_to_sequences(sample_review)
pad = pad_sequences(seq, maxlen=max_length, padding='post')

prediction = model.predict(pad)

print("Positive Review" if prediction[0][0] > 0.5 else "Negative Review")
