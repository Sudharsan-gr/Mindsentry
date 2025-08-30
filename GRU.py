import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---- Load data ----
data = np.load("video_emotion.npz")
X_train, X_val = data["X_train"], data["X_val"]
y_train, y_val = data["y_train"], data["y_val"]
label_classes = data["label_classes"]

num_classes = len(label_classes)  # Number of emotion classes

print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
print(f"Sequence length: {X_train.shape[1]}, Embedding dimension: {X_train.shape[2]}")
print(f"Number of classes: {num_classes}")

# ---- Build GRU model ----
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.4),
    GRU(64),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---- Train ----
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=150,
    callbacks=[early_stop]
)

# ---- Save model ----
model.save("gru_model.h5")
print("Model saved as 'gru_model.h5'.")

# ---- Usage example: predicting on validation data ----
predictions = model.predict(X_val)
predicted_classes = np.argmax(predictions, axis=1)

# Optional: print class names for first 5 predictions
for i in range(5):
    print(f"True: {label_classes[y_val[i]]}, Predicted: {label_classes[predicted_classes[i]]}")
