import os
import cv2
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns

# Paths and parameters (update as needed)
CNN_MODEL_PATH = r"C:\Users\grsud\Desktop\MIND SENTRY\models\mobilenetv2_fer2013_feature_extractor.h5"
GRU_MODEL_PATH = r"C:\Users\grsud\Desktop\MIND SENTRY\best_gru_model_20250907_100712.keras"  # Adjust if needed
TEST_DATA_DIR = r"C:\Users\grsud\Desktop\MIND SENTRY\Data\GRUEqual split\test"

SEQ_LEN = 30
FRAME_SKIP = 10
IMG_SIZE = 224
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.6

# Load models
print("Loading models...")
cnn_model = load_model(CNN_MODEL_PATH)
gru_model = load_model(GRU_MODEL_PATH, custom_objects={"Attention": tf.keras.layers.Layer})  # use your actual Attention class

# Emotion classes from test folder names
emotion_classes = sorted([d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d))])
print(f"Detected emotion classes: {emotion_classes}")

def extract_frames(video_path, frame_skip=FRAME_SKIP, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (img_size, img_size))
            processed = preprocess_input(resized)
            frames.append(processed)
        frame_count += 1
    cap.release()
    return frames

def frames_to_embeddings(frames, cnn_model, batch_size=BATCH_SIZE):
    embeddings = []
    for i in range(0, len(frames), batch_size):
        batch = np.array(frames[i:i+batch_size])
        emb_batch = cnn_model.predict(batch, verbose=0)
        embeddings.extend(emb_batch)
    return np.array(embeddings)

def predict_video_emotion(video_path):
    frames = extract_frames(video_path)
    if len(frames) == 0:
        return None, 0.0
    if len(frames) < SEQ_LEN:
        last_frame = frames[-1]
        while len(frames) < SEQ_LEN:
            frames.append(last_frame)
    elif len(frames) > SEQ_LEN:
        frames = frames[:SEQ_LEN]
    embeddings = frames_to_embeddings(frames, cnn_model)
    seq = embeddings[np.newaxis, ...]
    preds = gru_model.predict(seq, verbose=0)[0]
    max_prob = np.max(preds)
    max_idx = np.argmax(preds)
    predicted_emotion = emotion_classes[max_idx] if max_prob >= CONFIDENCE_THRESHOLD else "uncertain"
    return predicted_emotion, max_prob

emotion_counts = Counter()
video_results = []

print("Starting testing on video dataset...\n")

total_videos = 0
for emotion in emotion_classes:
    emotion_folder = os.path.join(TEST_DATA_DIR, emotion)
    video_files = [f for f in os.listdir(emotion_folder) if f.lower().endswith(('.mp4','.avi'))]
    print(f"Processing emotion '{emotion}' with {len(video_files)} videos...\n")
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(emotion_folder, video_file)
        predicted_emotion, confidence = predict_video_emotion(video_path)
        print(f"[{i}/{len(video_files)}] {video_file} | True: {emotion} | Predicted: {predicted_emotion} | Confidence: {confidence:.2f}")
        emotion_counts[predicted_emotion] += 1
        video_results.append((video_file, emotion, predicted_emotion, confidence))
        total_videos += 1

print("\n\n=== Final Prediction Counts ===")
for emotion, count in emotion_counts.most_common():
    print(f"{emotion:12s}: {count}")

print(f"\nTotal videos processed: {total_videos}")

# Visualization of predicted emotion distribution
plt.figure(figsize=(10,6))
sns.countplot(x=[r[2] for r in video_results], order=[e for e in emotion_classes] + ["uncertain"])
plt.title("Distribution of Predicted Emotions on Test Videos")
plt.xlabel("Predicted Emotion")
plt.ylabel("Number of Videos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("predicted_emotion_distribution.png")
plt.show()

print("\nTesting complete. Results saved to 'predicted_emotion_distribution.png'.")
