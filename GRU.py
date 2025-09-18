import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import os

# -----------------------
# Load Data
# -----------------------
print("Loading embeddings...")
data = np.load(r"C:\Users\grsud\Desktop\MIND SENTRY\Presentations\video_emotion.npz")

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
emotion_classes = list(data["label_classes"])

print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Classes: {emotion_classes}")
print(f"Label distribution (train): {np.bincount(y_train)}")

# -----------------------
# Compute Class Weights
# -----------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# -----------------------
# Build GRU Model
# -----------------------
def build_gru_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # First GRU block
    x = layers.Dropout(0.2)(inputs)
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)

    # Second GRU block
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)

    # Multi-head attention
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)

model = build_gru_model((X_train.shape[1], X_train.shape[2]), len(emotion_classes))

# -----------------------
# Compile
# -----------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------
# Callbacks
# -----------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
checkpoint_path = f"best_gru_model_{timestamp}.keras"

callbacks_list = [
    callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )
]

# -----------------------
# Train
# -----------------------
print("\nStarting training with class weights...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,   # ✅ use class weights
    callbacks=callbacks_list,
    verbose=1
)

# Save training history
history_path = f"gru_history_{timestamp}.json"
with open(history_path, "w") as f:
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    json.dump(history_dict, f, indent=2)
print(f"Training history saved to: {history_path}")

# -----------------------
# Load Best Model & Evaluate
# -----------------------
print("\nLoading best model for evaluation...")
best_model = tf.keras.models.load_model(checkpoint_path)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# -----------------------
# Predictions & Confusion Matrix
# -----------------------
print("Generating predictions...")
y_pred = best_model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=emotion_classes, yticklabels=emotion_classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - Test Accuracy: {test_acc:.2%}")
plt.tight_layout()
plt.savefig(f"gru_confusion_matrix_{timestamp}.png", dpi=150)
plt.show()

# -----------------------
# Training Curves
# -----------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(f"gru_training_curves_{timestamp}.png", dpi=150)
plt.show()

# -----------------------
# Classification Report
# -----------------------
print("\nClassification Report:")
report = classification_report(y_test, y_pred_classes, target_names=emotion_classes, digits=4)
print(report)

with open(f"classification_report_{timestamp}.txt", "w") as f:
    f.write(report)

# -----------------------
# Prediction Distribution
# -----------------------
unique, counts = np.unique(y_pred_classes, return_counts=True)
print("\nPrediction distribution:")
for cls, count in zip(unique, counts):
    print(f"{emotion_classes[cls]}: {count} ({count/len(y_pred_classes):.2%})")
