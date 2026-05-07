# =========================================
# 1. Import Libraries
# =========================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================================
# 2. Load CSV Dataset
# =========================================
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# =========================================
# 3. CLEAN DATA (VERY IMPORTANT)
# =========================================

# Convert everything to numeric (handles errors)
train_df = train_df.apply(pd.to_numeric, errors='coerce')
test_df = test_df.apply(pd.to_numeric, errors='coerce')

# Replace NaN values with 0
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# =========================================
# 4. Separate Features and Labels
# =========================================
X = train_df.drop("label", axis=1).values.astype("float32")
y = train_df["label"].values.astype("int32")

X_test = test_df.drop("label", axis=1).values.astype("float32")
y_test = test_df["label"].values.astype("int32")

# =========================================
# 5. Normalize (0–255 → 0–1)
# =========================================
X = X / 255.0
X_test = X_test / 255.0

# =========================================
# 6. Reshape for CNN
# =========================================
X = X.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# =========================================
# 7. DEBUG CHECK (IMPORTANT)
# =========================================
print("Any NaN in X:", np.isnan(X).any())
print("Any NaN in y:", np.isnan(y).any())

# =========================================
# 8. Train / Validation Split
# =========================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# =========================================
# 9. Build CNN Model
# =========================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# =========================================
# 10. Compile Model
# =========================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================
# 11. Train Model
# =========================================
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# =========================================
# 12. Evaluate Model
# =========================================
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\n✅ Test Accuracy:", test_acc)

# =========================================
# 13. Predictions
# =========================================
predictions = model.predict(X_test)

# =========================================
# 14. Class Labels
# =========================================
class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# =========================================
# 15. Show Prediction Example
# =========================================
index = 10  # change this to see different samples

plt.imshow(X_test[index].reshape(28,28), cmap='gray')
# plt.title(f"Predicted: {class_names[np.argmax(predictions[index])]}")
plt.title(f"Predicted: {class_names[np.argmax(predictions[index])]} ({np.max(predictions[index]) * 100:.2f}%)")
plt.axis('off')
plt.show()