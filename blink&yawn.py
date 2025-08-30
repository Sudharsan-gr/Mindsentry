import cv2
import numpy as np
from deepface import DeepFace
from collections import deque, Counter
import torch
import torch.nn as nn

# ----------- Simple GRU model for smoothing emotion predictions -----------
class EmotionGRUSmoother(nn.Module):
    def __init__(self, input_size=7, hidden_size=16, num_layers=1, num_classes=7):
        super(EmotionGRUSmoother, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        # x: batch_size=1, seq_len, input_size
        out, _ = self.gru(x)
        out = out[:, -1, :]  # take last time step
        out = self.fc(out)
        return self.softmax(out)

# Emotion labels used by DeepFace
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load GRU model (here just randomly initialized for demo, replace with trained weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gru_model = EmotionGRUSmoother().to(device)
gru_model.eval()

# Buffer to hold recent emotion probability vectors
SEQ_LEN = 10
emotion_prob_buffer = deque(maxlen=SEQ_LEN)

cap = cv2.VideoCapture(0)

print("Starting video capture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Use DeepFace to analyze emotion probabilities (returns dict)
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]  # sometimes returns a list

        emotion_probs = analysis['emotion']  # dict of probs

        # Convert dict to numpy array ordered by EMOTION_LABELS
        prob_vector = np.array([emotion_probs.get(label, 0) for label in EMOTION_LABELS], dtype=np.float32)
        prob_vector /= prob_vector.sum()  # normalize

        # Add to buffer
        emotion_prob_buffer.append(prob_vector)

        # If we have enough frames, feed to GRU to get smoothed emotion
        if len(emotion_prob_buffer) == SEQ_LEN:
            seq_tensor = torch.tensor([emotion_prob_buffer], dtype=torch.float32).to(device)  # shape: (1, seq_len, 7)
            with torch.no_grad():
                smoothed_probs = gru_model(seq_tensor).cpu().numpy().flatten()
            dominant_emotion = EMOTION_LABELS[np.argmax(smoothed_probs)]
        else:
            dominant_emotion = analysis['dominant_emotion']

    except Exception as e:
        dominant_emotion = "neutral"

    # Show emotion on frame
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Emotion Detection with GRU Smoothing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
