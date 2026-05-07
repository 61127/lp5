import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# LOAD DATA (FIXED)
# -------------------------
data = pd.read_csv(
    "IMDB Dataset.csv",
)

# -------------------------
# CLEAN TEXT
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["review"] = data["review"].apply(clean_text)
data["sentiment"] = data["sentiment"].map({"positive":1, "negative":0})

# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["review"], data["sentiment"],
    test_size=0.2,
    random_state=42
)

# -------------------------
# TOKENIZATION
# -------------------------
vocab_size = 40000
max_len = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train) ##word to index mapping

X_train_seq = tokenizer.texts_to_sequences(X_train) ##sentence to sequence of numbers
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

# -------------------------
# IMPROVED MODEL
# -------------------------
model = Sequential([
    Embedding(vocab_size, 128),

    Bidirectional(LSTM(32, dropout=0.3)),

    Dense(64, activation='relu'),
    Dropout(0.6),

    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# -------------------------
# EARLY STOPPING
# -------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# -------------------------
# TRAIN
# -------------------------
history = model.fit(
    X_train_pad,
    y_train,
    epochs=6,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop]
)

# -------------------------
# EVALUATE
# -------------------------
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# -------------------------
# PREDICTION
# -------------------------
def predict_review(review):
    review = clean_text(review)
    seq = tokenizer.texts_to_sequences([review])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)[0][0]
    return "Positive" if pred > 0.5 else "Negative"

print("What a boring day!", predict_review("What a boring day!"))
print("I absolutely loved this movie, it was fantastic!", predict_review("I absolutely loved this movie, it was fantastic!"))
print("The service was terrible and very disappointing.", predict_review("The service was terrible and very disappointing."))