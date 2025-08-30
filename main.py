import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time


# Load models
cnn_model = tf.keras.models.load_model("C:\\Users\\grsud\\Desktop\\MIND SENTRY\\models\\mobilenetv2_fer2013_feature_extractor.h5")
gru_model = tf.keras.models.load_model('C:\\Users\\grsud\\Desktop\\MIND SENTRY\\gru_model.h5')


emotion_classes = ["happy", "sad", "angry", "disgust", "fearful", "surprise", "neutral"]
stress_emotions = {"sad", "angry", "disgust", "fearful", "neutral"}


def stress_probability(pred_probs):
    return sum(p for p, e in zip(pred_probs, emotion_classes) if e in stress_emotions)


SEQ_LEN = 30
FRAME_SKIP = 10
IMG_SIZE = 224
BUFFER_LIMIT = 540  # total frames embeddings to capture (3 min @ every 10th frame)


embedding_buffer = deque(maxlen=SEQ_LEN)
all_embeddings = []


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


start_time = time.time()
frame_count = 0


print("Starting 3-minute stress detection...")


while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame_count += 1


    # Convert to gray for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


    if len(faces) > 0:
        x, y, w, h = faces[0]
        # Validate bounding box bounds
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
        face_img = frame[y0:y1, x0:x1]
        cv2.imshow('Face', face_img)
    else:
        # No face detected, show empty window or original frame
        cv2.imshow('Face', frame)


    # Process every 10th frame
    if frame_count % FRAME_SKIP == 0 and (len(all_embeddings) * SEQ_LEN + len(embedding_buffer)) < BUFFER_LIMIT:
        if len(faces) > 0:
            img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            emb = cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            embedding_buffer.append(emb)
            if len(embedding_buffer) == SEQ_LEN:
                all_embeddings.append(np.array(embedding_buffer))
                embedding_buffer.clear()


    elapsed = time.time() - start_time
    if elapsed > 60 or len(all_embeddings) >= (BUFFER_LIMIT // SEQ_LEN):
        break


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


print("Analyzing captured data...")


stress_probs = []

# To accumulate total emotion probabilities for percentage calculation
total_emotion_probs = np.zeros(len(emotion_classes))

for seq in all_embeddings:
    seq = seq[np.newaxis, ...]
    pred_probs = gru_model.predict(seq, verbose=0)[0]

    # Accumulate emotion probabilities
    total_emotion_probs += pred_probs

    sp = stress_probability(pred_probs)
    stress_probs.append(sp)

# Calculate average emotion probabilities percentage
if len(all_embeddings) > 0:
    avg_emotion_probs = total_emotion_probs / len(all_embeddings)
    print("\nAverage Emotion Probabilities:")
    for emotion, prob in zip(emotion_classes, avg_emotion_probs):
        print(f"{emotion}: {prob*100:.2f}%")
else:
    print("No embeddings captured.")

avg_stress_prob = np.mean(stress_probs) if stress_probs else 0
stress_status = "Stress" if avg_stress_prob > 0.5 else "No Stress"


print(f"\n3-minute Stress Analysis Complete.")
print(f"Average Stress Probability: {avg_stress_prob:.3f}")
print(f"Final Stress Status: {stress_status}")
