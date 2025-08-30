import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Config ----
DATA_DIR = "C:\\Users\\grsud\\Desktop\\MIND SENTRY\\Data\\GRU training"  # folder containing emotion subfolders
CNN_MODEL_PATH = "C:\\Users\\grsud\\Desktop\\MIND SENTRY\\models\\mobilenetv2_fer2013_feature_extractor.h5"
FRAME_SKIP = 10
IMG_SIZE = 224
SEQ_LEN = 30  # fixed sequence length for padding/truncating

# ---- Extract frames from video ----
def extract_frames(video_path, frame_skip=10, img_size=224):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame, (img_size, img_size))
            normalized = resized.astype('float32') / 255.0
            frames.append(normalized)
        frame_count += 1
    cap.release()
    return frames

# ---- Convert frames to embeddings using CNN ----
def frames_to_embeddings(frames, cnn_model):
    embeddings = []
    for frame in frames:
        x = np.expand_dims(frame, axis=0)
        emb = cnn_model.predict(x, verbose=0)
        embeddings.append(emb[0])
    return np.array(embeddings)

# ---- Prepare dataset from folders ----
def prepare_dataset(data_dir, cnn_model, frame_skip=10, img_size=224, seq_len=30):
    X_sequences = []
    y_labels = []
    emotions = sorted(os.listdir(data_dir))  # keep consistent ordering

    for emotion in emotions:
        emotion_folder = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        print(f"Processing emotion: {emotion}")
        for fname in os.listdir(emotion_folder):
            if not fname.endswith('.mp4'):
                continue
            video_path = os.path.join(emotion_folder, fname)
            frames = extract_frames(video_path, frame_skip, img_size)
            if len(frames) == 0:
                print(f"Warning: No frames extracted for {video_path}")
                continue
            embeddings = frames_to_embeddings(frames, cnn_model)
            X_sequences.append(embeddings)
            y_labels.append(emotion)

    print("Padding/truncating sequences to fixed length:", seq_len)
    X_padded = pad_sequences(
        X_sequences,
        maxlen=seq_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )
    return X_padded, y_labels

# ---- Main Script ----
if __name__ == "__main__":
    print("Loading CNN feature extractor...")
    feature_extractor = tf.keras.models.load_model(CNN_MODEL_PATH)

    print("Preparing dataset...")
    X_sequences, y_labels = prepare_dataset(DATA_DIR, feature_extractor, FRAME_SKIP, IMG_SIZE, SEQ_LEN)

    print(f"Loaded {len(X_sequences)} video sequences.")

    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    print(f"X_data shape: {X_sequences.shape}")  # Should be (num_videos, SEQ_LEN, embedding_dim)
    print(f"y_data shape: {len(y_encoded)}")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_sequences, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Save processed data for GRU training
    np.savez("video_emotion.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, label_classes=le.classes_)
    print("Data saved as 'video_emotion.npz'. Ready for GRU training!")
