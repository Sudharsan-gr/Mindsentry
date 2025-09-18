import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm  # For progress bars

# Config
DATA_SPLIT_DIRS = {
    "train": r"C:\Users\grsud\Desktop\MIND SENTRY\Data\CREMA\train",
    "val": r"C:\Users\grsud\Desktop\MIND SENTRY\Data\CREMA\val",
    "test": r"C:\Users\grsud\Desktop\MIND SENTRY\Data\CREMA\test"
}

CNN_MODEL_PATH = r"C:\Users\grsud\Desktop\MIND SENTRY\models\mobilenetv2_fer2013_feature_extractor.h5"
FRAME_SKIP = 2
IMG_SIZE = 224
SEQ_LEN = 30
MIN_FRAMES = 5  # Minimum frames required for a valid video

# Enhanced frame extraction with quality checks
def extract_frames(video_path, frame_skip=10, img_size=224):
    """Extract frames with quality validation"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < MIN_FRAMES * frame_skip:
        print(f"Warning: Video too short: {video_path} ({total_frames} frames)")
        cap.release()
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            # Check frame quality
            if frame is not None and frame.size > 0:
                # Apply CLAHE for better contrast
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
                
                # Resize and normalize
                resized = cv2.resize(frame, (img_size, img_size))
                normalized = resized.astype('float32') / 255.0
                frames.append(normalized)
                
        frame_count += 1
    
    cap.release()
    
    # Validate extracted frames
    if len(frames) < MIN_FRAMES:
        print(f"Warning: Too few frames extracted from {video_path} ({len(frames)} frames)")
        return None
        
    return frames

# Data augmentation for frames
def augment_frames(frames, augment_prob=0.5):
    """Apply random augmentations to frames"""
    if np.random.random() > augment_prob:
        return frames
        
    augmented = []
    for frame in frames:
        aug_frame = frame.copy()
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            aug_frame = cv2.flip(aug_frame, 1)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            aug_frame = np.clip(aug_frame * factor, 0, 1)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            aug_frame = np.clip((aug_frame - 0.5) * factor + 0.5, 0, 1)
            
        augmented.append(aug_frame)
    
    return augmented

# Convert frames to CNN embeddings with batch processing
def frames_to_embeddings(frames, cnn_model, batch_size=32):
    """Convert frames to embeddings using batch processing for efficiency"""
    embeddings = []
    
    # Process in batches
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_array = np.array(batch)
        batch_embeddings = cnn_model.predict(batch_array, verbose=0)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

# Prepare dataset with augmentation option
def prepare_dataset(data_dir, cnn_model, frame_skip=10, img_size=224, seq_len=30, 
                   augment=False, augment_prob=0.5):
    """Prepare dataset with optional augmentation"""
    X_sequences = []
    y_labels = []
    emotions = sorted(os.listdir(data_dir))
    
    # Statistics
    valid_videos = 0
    skipped_videos = 0
    
    print(f"\nProcessing {data_dir}")
    print(f"Emotions found: {emotions}")
    
    for emotion in emotions:
        emotion_folder = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_folder):
            continue
            
        video_files = [f for f in os.listdir(emotion_folder) if f.lower().endswith('.mp4')]
        print(f"\nProcessing {emotion}: {len(video_files)} videos")
        
        # Progress bar for each emotion
        for fname in tqdm(video_files, desc=f"  {emotion}"):
            video_path = os.path.join(emotion_folder, fname)
            
            # Extract frames
            frames = extract_frames(video_path, frame_skip, img_size)
            
            if frames is None:
                skipped_videos += 1
                continue
            
            # Apply augmentation if training data
            if augment and 'train' in data_dir.lower():
                frames = augment_frames(frames, augment_prob)
            
            # Convert to embeddings
            embeddings = frames_to_embeddings(frames, cnn_model)
            
            if len(embeddings) > 0:
                X_sequences.append(embeddings)
                y_labels.append(emotion)
                valid_videos += 1
                
                # For training, add augmented version
                if augment and 'train' in data_dir.lower() and np.random.random() > 0.5:
                    aug_frames = augment_frames(frames, augment_prob=0.8)
                    aug_embeddings = frames_to_embeddings(aug_frames, cnn_model)
                    X_sequences.append(aug_embeddings)
                    y_labels.append(emotion)
    
    print(f"\nDataset Summary for {data_dir}:")
    print(f"  Valid videos: {valid_videos}")
    print(f"  Skipped videos: {skipped_videos}")
    print(f"  Total sequences: {len(X_sequences)}")
    
    if len(X_sequences) == 0:
        print(f"ERROR: No valid sequences found in {data_dir}")
        return None, None
    
    # Pad sequences
    X_padded = pad_sequences(
        X_sequences,
        maxlen=seq_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )
    
    return X_padded, y_labels

if __name__ == "__main__":
    print("="*50)
    print("Enhanced Video Emotion Embedding Generator")
    print("="*50)
    
    print("\nLoading CNN model...")
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print(f"Model loaded: {CNN_MODEL_PATH}")
    
    # Process each split with augmentation for training
    print("\n" + "="*50)
    print("Processing Training Data (with augmentation)...")
    X_train, y_train = prepare_dataset(
        DATA_SPLIT_DIRS["train"], 
        cnn_model, 
        FRAME_SKIP, 
        IMG_SIZE, 
        SEQ_LEN,
        augment=True,  # Enable augmentation for training
        augment_prob=0.5
    )
    
    print("\n" + "="*50)
    print("Processing Validation Data...")
    X_val, y_val = prepare_dataset(
        DATA_SPLIT_DIRS["val"], 
        cnn_model, 
        FRAME_SKIP, 
        IMG_SIZE, 
        SEQ_LEN,
        augment=False  # No augmentation for validation
    )
    
    print("\n" + "="*50)
    print("Processing Test Data...")
    X_test, y_test = prepare_dataset(
        DATA_SPLIT_DIRS["test"], 
        cnn_model, 
        FRAME_SKIP, 
        IMG_SIZE, 
        SEQ_LEN,
        augment=False  # No augmentation for test
    )
    
    # Check if all datasets are valid
    if X_train is None or X_val is None or X_test is None:
        print("\nERROR: Failed to process one or more datasets. Exiting.")
        exit(1)
    
    # Encode labels consistently
    print("\n" + "="*50)
    print("Encoding labels...")
    le = LabelEncoder()
    all_labels = y_train + y_val + y_test
    le.fit(all_labels)
    
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    
    print(f"Label classes: {list(le.classes_)}")
    
    # Print final statistics
    print("\n" + "="*50)
    print("FINAL DATASET STATISTICS")
    print("="*50)
    print(f"Training:   {X_train.shape[0]} sequences, shape: {X_train.shape}")
    print(f"Validation: {X_val.shape[0]} sequences, shape: {X_val.shape}")
    print(f"Test:       {X_test.shape[0]} sequences, shape: {X_test.shape}")
    
    # Print class distribution
    print("\nClass Distribution:")
    for emotion in le.classes_:
        train_count = np.sum(y_train_enc == le.transform([emotion])[0])
        val_count = np.sum(y_val_enc == le.transform([emotion])[0])
        test_count = np.sum(y_test_enc == le.transform([emotion])[0])
        print(f"  {emotion:10s}: Train={train_count:3d}, Val={val_count:3d}, Test={test_count:3d}")
    
    # Save with additional metadata
    output_file = "embeddings_crema_gru.npz"
    np.savez(output_file,
             X_train=X_train, y_train=y_train_enc,
             X_val=X_val, y_val=y_val_enc,
             X_test=X_test, y_test=y_test_enc,
             label_classes=le.classes_,
             frame_skip=FRAME_SKIP,
             seq_len=SEQ_LEN,
             img_size=IMG_SIZE)
    
    print(f"\n✅ Saved enhanced embeddings to '{output_file}'")
    print("Ready for GRU training!")
