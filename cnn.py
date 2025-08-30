import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Dataset paths ---
train_dir = r'C:\Users\grsud\Desktop\MIND SENTRY\Data\train'
test_dir = r'C:\Users\grsud\Desktop\MIND SENTRY\Data\test'

# --- Parameters ---
batch_size = 64
img_size = (224, 224)  # MobileNetV2 input size
initial_epochs = 25
fine_tune_epochs = 15
total_epochs = initial_epochs + fine_tune_epochs

# --- Load datasets no mapping (to keep class_names) ---
train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.15,
    subset='training',
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.15,
    subset='validation',
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

test_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
)

class_names = train_ds_raw.class_names
num_classes = len(class_names)
print(f"Class names: {class_names}")

# --- Normalize pixels 0-255 to 0-1 ---
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds_raw.map(lambda x, y: (normalization_layer(x), y))

# --- Prefetch for performance ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# --- Define model with explicit Input layer ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size[0], img_size[1], 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False  # Freeze for initial training

inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
x = base_model(inputs, training=False)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# --- Compile model ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Callbacks ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# --- Initial training ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=[early_stop, reduce_lr]
)

# --- Fine-tuning ---
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50  # Freeze except last 50 layers

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with smaller learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continue fine-tuning
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[early_stop, reduce_lr]
)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

# --- Save full trained model ---
model.save('trained_mobilenetv2_fer2013.h5')

# --- Save feature extractor model ---
# Call model once to ensure build
_ = model(next(iter(train_ds))[0])

feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-3].output  # Output of Dense(128) before Dropout and final Dense
)
feature_model.save('mobilenetv2_fer2013_feature_extractor.h5')
