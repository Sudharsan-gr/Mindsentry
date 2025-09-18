import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# Custom Attention Layer (must match training)
# -----------------------
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        alpha = tf.keras.backend.softmax(e, axis=1)
        context = x * alpha
        return tf.keras.backend.sum(context, axis=1)

# -----------------------
# Load Models
# -----------------------
print("Loading models...")
cnn_model = tf.keras.models.load_model(
    r"C:\Users\grsud\Desktop\MIND SENTRY\models\mobilenetv2_fer2013_feature_extractor.h5"
)
gru_model = tf.keras.models.load_model(
    r"C:\Users\grsud\Desktop\MIND SENTRY\GRU hap sad.keras",  # Fixed path
    custom_objects={"Attention": Attention},
)

# Load label classes from saved embeddings to ensure consistency
data = np.load(r"C:\Users\grsud\Desktop\MIND SENTRY\embeddings_equal.npz")
emotion_classes = list(data["label_classes"])
print(f"Loaded emotion classes: {emotion_classes}")

# Enhanced Weighted Stress Mapping with validation
emotion_to_stress_weight = {
    "sad": 0.6,      # Increased from 0.6
    "angry": 0.85,   # Slightly reduced from 0.9
    "disgust": 0.65, # Slightly reduced
    "fear": 0.95,    # Slightly reduced from 1.0
    "happy": 0,   # Negative weight for stress reduction
    "surprised": 0.3, # Slight increase
    "neutral": 0.1,  # Small baseline stress
}

# Validate all emotions are mapped
for emotion in emotion_classes:
    if emotion not in emotion_to_stress_weight:
        print(f"Warning: {emotion} not in stress weight mapping, setting to 0.5")
        emotion_to_stress_weight[emotion] = 0.5

def temperature_softmax(logits, temperature=2.0):
    """Apply temperature scaling to soften probability distribution"""
    # Convert probabilities back to logits (approximate)
    epsilon = 1e-10
    logits = np.log(np.clip(logits, epsilon, 1.0))
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Convert back to probabilities
    return softmax(scaled_logits)

def smooth_predictions(predictions_buffer, window_size=5):
    """Smooth predictions over time using moving average"""
    if len(predictions_buffer) < window_size:
        return predictions_buffer[-1] if predictions_buffer else np.zeros(len(emotion_classes))
    
    # Take the last window_size predictions and average them
    recent_preds = np.array(list(predictions_buffer)[-window_size:])
    return np.mean(recent_preds, axis=0)

def stress_probability(pred_probs, use_weighted=True):
    """Map emotion probabilities to stress probability with normalization."""
    if use_weighted:
        stress_score = sum(p * emotion_to_stress_weight.get(e, 0.5) 
                          for p, e in zip(pred_probs, emotion_classes))
        # Normalize to [0, 1] range
        return np.clip((stress_score + 0.2) / 1.2, 0, 1)
    else:
        # Simple stress calculation (fear + angry + sad + disgust)
        stress_emotions = ["fear", "angry", "sad", "disgust"]
        return sum(pred_probs[emotion_classes.index(e)] 
                  for e in stress_emotions if e in emotion_classes)

# -----------------------
# Enhanced Parameters
# -----------------------
SEQ_LEN = 30          # Sequence length for GRU
FRAME_SKIP = 10       # Process every 10th frame
IMG_SIZE = 224        # CNN input size
DURATION = 180        # 3 minutes in seconds
MIN_FACE_SIZE = 50    # Minimum face size to process
CONFIDENCE_THRESHOLD = 0.3  # Reduced threshold for more balanced display
TEMPERATURE = 3.0     # Temperature for softmax scaling (higher = more distributed)
SMOOTHING_WINDOW = 7  # Window size for temporal smoothing

# Buffers and tracking
embedding_buffer = deque(maxlen=SEQ_LEN)
predictions_buffer = deque(maxlen=SMOOTHING_WINDOW)
all_sequences = []
face_quality_scores = []
timestamp_sequences = []

# Enhanced face detection with DNN option (more accurate)
use_dnn = False  # Set to True if you have the DNN model files
if use_dnn:
    # Load DNN face detector (more accurate but slower)
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_dnn(frame, conf_threshold=0.5):
    """DNN-based face detection (more accurate)"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append((x, y, x1-x, y1-y))
    return faces

def preprocess_face(face_img):
    """Enhanced face preprocessing"""
    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    face_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # Resize and normalize
    img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

start_time = time.time()
frame_count = 0
valid_frame_count = 0

print(f"Starting {DURATION}-second stress detection...")
print("Press 'q' to quit early")

# Real-time display variables
stress_history = deque(maxlen=20)
emotion_history = deque(maxlen=10)
current_stress = 0.0
current_emotion_probs = np.zeros(len(emotion_classes))
dominant_emotion = "neutral"

# Create display window
cv2.namedWindow("Stress Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    display_frame = frame.copy()
    
    # Face detection
    if use_dnn:
        faces = detect_face_dnn(frame)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                             minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
    
    face_detected = len(faces) > 0
    face_img = None
    
    if face_detected:
        # Use largest face
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        
        x, y, w, h = faces[0]
        # Add padding around face
        padding = int(0.1 * min(w, h))
        x0 = max(x - padding, 0)
        y0 = max(y - padding, 0)
        x1 = min(x + w + padding, frame.shape[1])
        y1 = min(y + h + padding, frame.shape[0])
        
        face_img = frame[y0:y1, x0:x1]
        
        # Draw face rectangle
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calculate face quality score (based on size and position)
        face_quality = min(w, h) / MIN_FACE_SIZE
        face_quality_scores.append(face_quality)
    
    # Process frame for embedding
    if frame_count % FRAME_SKIP == 0 and face_img is not None:
        try:
            img = preprocess_face(face_img)
            emb = cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            embedding_buffer.append(emb)
            valid_frame_count += 1
            
            # When we have a full sequence
            if len(embedding_buffer) == SEQ_LEN:
                seq_array = np.array(embedding_buffer)
                all_sequences.append(seq_array)
                timestamp_sequences.append(time.time() - start_time)
                
                # Get prediction for real-time display
                seq_input = seq_array[np.newaxis, ...]
                raw_pred_probs = gru_model.predict(seq_input, verbose=0)[0]
                
                # Apply temperature scaling to make predictions less confident
                pred_probs = temperature_softmax(raw_pred_probs, TEMPERATURE)
                
                # Add to predictions buffer for smoothing
                predictions_buffer.append(pred_probs)
                
                # Apply temporal smoothing
                current_emotion_probs = smooth_predictions(predictions_buffer, SMOOTHING_WINDOW)
                
                # Update current stress
                current_stress = stress_probability(current_emotion_probs)
                stress_history.append(current_stress)
                
                # Get dominant emotion (but don't rely solely on it)
                max_idx = np.argmax(current_emotion_probs)
                dominant_emotion = emotion_classes[max_idx]
                
                # Clear buffer for next sequence (with overlap)
                # Keep last 10 embeddings for temporal continuity
                embedding_buffer = deque(list(embedding_buffer)[-10:], maxlen=SEQ_LEN)
                
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    # Display real-time information
    info_y = 30
    cv2.putText(display_frame, f"Time: {int(time.time() - start_time)}s / {DURATION}s", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += 25
    
    if face_detected:
        cv2.putText(display_frame, "Face: Detected", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Face: Not Found", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    info_y += 25
    
    # Display stress level with color coding
    stress_color = (0, 255, 0)  # Green
    if current_stress > 0.7:
        stress_color = (0, 0, 255)  # Red
    elif current_stress > 0.4:
        stress_color = (0, 165, 255)  # Orange
    
    cv2.putText(display_frame, f"Stress: {current_stress:.2%}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stress_color, 2)
    info_y += 25
    
    # Display top 3 emotions with their probabilities
    if len(current_emotion_probs) > 0:
        # Sort emotions by probability
        emotion_prob_pairs = list(zip(emotion_classes, current_emotion_probs))
        emotion_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        cv2.putText(display_frame, "Top Emotions:", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        info_y += 20
        
        # Display top 3 emotions
        for i, (emotion, prob) in enumerate(emotion_prob_pairs[:3]):
            if prob > 0.05:  # Only show if probability > 5%
                color = (0, 255, 255) if i == 0 else (200, 200, 200)
                cv2.putText(display_frame, f"{emotion}: {prob:.1%}", 
                           (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                info_y += 18
    
    cv2.putText(display_frame, f"Valid Frames: {valid_frame_count}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw stress bar
    bar_width = int(200 * current_stress)
    cv2.rectangle(display_frame, (10, frame.shape[0] - 40), 
                 (10 + bar_width, frame.shape[0] - 20), stress_color, -1)
    cv2.rectangle(display_frame, (10, frame.shape[0] - 40), 
                 (210, frame.shape[0] - 20), (255, 255, 255), 2)
    
    cv2.imshow("Stress Detection", display_frame)
    
    # Check exit conditions
    elapsed = time.time() - start_time
    if elapsed > DURATION:
        print(f"\nCompleted {DURATION}-second recording")
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nRecording stopped by user")
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------
# Final Analysis
# -----------------------
print("\n" + "="*50)
print("ANALYZING CAPTURED DATA...")
print("="*50)

if len(all_sequences) == 0:
    print("ERROR: No valid sequences captured. Please ensure:")
    print("- Your face is visible to the camera")
    print("- Adequate lighting is available")
    print("- Camera is working properly")
    exit(1)

# Process all sequences for final analysis
stress_probs = []
emotion_counts = {e: 0 for e in emotion_classes}
total_emotion_probs = np.zeros(len(emotion_classes))
confidence_scores = []

for i, seq in enumerate(all_sequences):
    seq_input = seq[np.newaxis, ...]
    raw_pred_probs = gru_model.predict(seq_input, verbose=0)[0]
    
    # Apply temperature scaling for final analysis too
    pred_probs = temperature_softmax(raw_pred_probs, TEMPERATURE)
    
    # Calculate confidence (entropy-based measure for distributed predictions)
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-10))
    max_entropy = np.log(len(emotion_classes))
    confidence = 1 - (entropy / max_entropy)  # Higher when less distributed
    confidence_scores.append(confidence)
    
    # Update emotion statistics
    total_emotion_probs += pred_probs
    
    # Count emotions that have significant probability (>15%)
    for idx, prob in enumerate(pred_probs):
        if prob > 0.15:
            emotion_counts[emotion_classes[idx]] += 1
    
    # Calculate stress
    sp = stress_probability(pred_probs)
    stress_probs.append(sp)

# Calculate final metrics
avg_emotion_probs = total_emotion_probs / len(all_sequences)
avg_stress = np.mean(stress_probs)
std_stress = np.std(stress_probs)
avg_confidence = np.mean(confidence_scores)

# Determine stress level category
if avg_stress < 0.3:
    stress_level = "LOW"
    stress_color_name = "Green"
elif avg_stress < 0.5:
    stress_level = "MILD"
    stress_color_name = "Yellow"
elif avg_stress < 0.7:
    stress_level = "MODERATE"
    stress_color_name = "Orange"
else:
    stress_level = "HIGH"
    stress_color_name = "Red"

# Print detailed results
print(f"\nTotal Sequences Analyzed: {len(all_sequences)}")
print(f"Average Model Confidence: {avg_confidence:.2%}")
print(f"Face Detection Quality: {np.mean(face_quality_scores):.2f}")

print("\n" + "-"*40)
print("EMOTION DISTRIBUTION (BALANCED):")
print("-"*40)
for emotion, prob in sorted(zip(emotion_classes, avg_emotion_probs), 
                           key=lambda x: x[1], reverse=True):
    bar = "█" * int(prob * 50)
    print(f"{emotion:10s}: {prob*100:5.1f}% {bar}")

print("\n" + "-"*40)
print("STRESS ANALYSIS:")
print("-"*40)
print(f"Average Stress Level: {avg_stress:.2%} ({stress_level})")
print(f"Stress Variability: ±{std_stress:.2%}")
print(f"Peak Stress: {np.max(stress_probs):.2%}")
print(f"Minimum Stress: {np.min(stress_probs):.2%}")

# Significant emotions analysis (emotions with >15% probability in sequences)
print("\n" + "-"*40)
print("SIGNIFICANT EMOTIONS FREQUENCY (>15% threshold):")
print("-"*40)
sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
for emotion, count in sorted_emotions:
    if count > 0:
        print(f"{emotion}: {count} sequences ({count/len(all_sequences)*100:.1f}%)")

# -----------------------
# Generate Enhanced Visualizations
# -----------------------
print("\nGenerating visualization reports...")

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))

# 1. Emotion Distribution Bar Chart (Updated)
ax1 = plt.subplot(2, 3, 1)
colors = ['red' if e in ['angry', 'fear', 'sad', 'disgust'] else 'green' if e == 'happy' else 'gray' 
          for e in emotion_classes]
bars = ax1.bar(emotion_classes, avg_emotion_probs * 100, color=colors, alpha=0.7)
ax1.set_ylabel("Probability (%)")
ax1.set_title("Balanced Emotion Distribution")
ax1.set_ylim(0, max(avg_emotion_probs * 100) + 5)
plt.xticks(rotation=45)
for bar, prob in zip(bars, avg_emotion_probs):
    height = bar.get_height()
    if height > 2:  # Only label if bar is tall enough
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=8)

# 2. Stress Timeline
ax2 = plt.subplot(2, 3, 2)
if timestamp_sequences:
    ax2.plot(timestamp_sequences[:len(stress_probs)], stress_probs, 
            marker="o", linestyle="-", color="red", alpha=0.6, label="Stress Level")
    ax2.axhline(0.5, color="blue", linestyle="--", alpha=0.5, label="Threshold")
    ax2.fill_between(timestamp_sequences[:len(stress_probs)], 
                     stress_probs, alpha=0.3, color="red")
else:
    ax2.plot(stress_probs, marker="o", linestyle="-", color="red", alpha=0.6)
    ax2.axhline(0.5, color="blue", linestyle="--", alpha=0.5)
ax2.set_xlabel("Time (seconds)" if timestamp_sequences else "Sequence Index")
ax2.set_ylabel("Stress Probability")
ax2.set_title("Stress Level Over Time")
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Stress Distribution Histogram
ax3 = plt.subplot(2, 3, 3)
ax3.hist(stress_probs, bins=20, color="orange", alpha=0.7, edgecolor='black')
ax3.axvline(avg_stress, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_stress:.2f}')
ax3.set_xlabel("Stress Probability")
ax3.set_ylabel("Frequency")
ax3.set_title("Stress Distribution")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Emotion Pie Chart (Updated)
ax4 = plt.subplot(2, 3, 4)
# Only show emotions with >5% to avoid cluttered pie chart
significant_emotions = [(e, p) for e, p in zip(emotion_classes, avg_emotion_probs) if p > 0.05]
if not significant_emotions:
    significant_emotions = list(zip(emotion_classes, avg_emotion_probs))

sig_emotions, sig_probs = zip(*significant_emotions)
colors_pie = plt.cm.Set3(range(len(sig_emotions)))
wedges, texts, autotexts = ax4.pie(sig_probs, labels=sig_emotions, 
                                    autopct=lambda pct: f'{pct:.1f}%' if pct > 8 else '',
                                    colors=colors_pie, startangle=90)
ax4.set_title("Emotion Composition (>5%)")

# 5. Model Confidence Scores
ax5 = plt.subplot(2, 3, 5)
ax5.plot(confidence_scores, marker="s", linestyle="-", color="purple", alpha=0.6)
ax5.set_xlabel("Sequence Index")
ax5.set_ylabel("Prediction Confidence")
ax5.set_title("Model Confidence Over Time")
ax5.set_ylim(0, 1)
ax5.grid(True, alpha=0.3)

# 6. Enhanced Summary Text Box
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
BALANCED EMOTION ANALYSIS
{'='*30}

Duration: {int(elapsed)} seconds
Sequences: {len(all_sequences)}
Valid Frames: {valid_frame_count}

STRESS LEVEL: {stress_level}
Average: {avg_stress:.1%}
Variation: ±{std_stress:.1%}

Top Emotions:
"""
for emotion, prob in sorted(zip(emotion_classes, avg_emotion_probs), 
                           key=lambda x: x[1], reverse=True)[:4]:
    if prob > 0.08:  # Only show significant emotions
        summary_text += f"• {emotion}: {prob*100:.1f}%\n"

summary_text += f"\nDistribution Quality:\n"
# Calculate distribution entropy as a measure of balance
entropy = -np.sum(avg_emotion_probs * np.log(avg_emotion_probs + 1e-10))
max_entropy = np.log(len(emotion_classes))
distribution_balance = entropy / max_entropy
summary_text += f"Balance Score: {distribution_balance:.2f}/1.0\n"

summary_text += f"\nRecommendation:\n"
if avg_stress > 0.7:
    summary_text += "High stress detected.\nConsider relaxation\ntechniques."
elif avg_stress > 0.5:
    summary_text += "Moderate stress levels.\nTake regular breaks."
elif avg_stress > 0.3:
    summary_text += "Mild stress present.\nMaintain awareness."
else:
    summary_text += "Low stress levels.\nGood emotional state."

ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle(f"Balanced Emotion Detection Analysis - {stress_level} STRESS", 
            fontsize=16, fontweight='bold')
plt.tight_layout()

# Save and show
timestamp = time.strftime("%Y%m%d_%H%M%S")
plt.savefig(f"balanced_emotion_analysis_{timestamp}.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Report saved as 'balanced_emotion_analysis_{timestamp}.png'")
print(f"\nFINAL VERDICT: {stress_level} STRESS ({avg_stress:.1%})")
print(f"Emotion Distribution Balance: {distribution_balance:.2f}/1.0")
