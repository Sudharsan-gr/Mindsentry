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
FRAME_SKIP = 5
IMG_SIZE = 224
SEQ_LEN = 30
MIN_FRAMES = 5  # Minimum frames required for a valid video

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.flv', '.avi', '.mov', '.mkv', '.wmv', '.m4v']

def get_video_files(directory):
    """Get all supported video files from a directory"""
    video_files = []
    if not os.path.exists(directory):
        print(f"Warning: Directory does not exist: {directory}")
        return video_files
    
    for file in os.listdir(directory):
        file_lower = file.lower()
        if any(file_lower.endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS):
            video_files.append(file)
    
    return video_files

def check_video_codec_support(video_path):
    """Check if video can be opened and has valid codec"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False, "Cannot read frames from video"
        
        return True, "Video is readable"
    
    except Exception as e:
        return False, f"Error checking video: {str(e)}"

# Enhanced frame extraction with quality checks and format support
def extract_frames(video_path, frame_skip=10, img_size=224):
    """Extract frames with quality validation and support for multiple formats"""
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File does not exist: {video_path}")
        return None
    
    # Check video codec support
    is_readable, message = check_video_codec_support(video_path)
    if not is_readable:
        print(f"Error: {video_path} - {message}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    successful_frames = 0
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # For FLV files, frame count might be unreliable, so we'll be more flexible
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext == '.flv':
            print(f"  Processing FLV file: {os.path.basename(video_path)}")
        
        # More flexible check for minimum frames (especially for FLV)
        if total_frames > 0 and total_frames < MIN_FRAMES * frame_skip:
            print(f"Warning: Video might be too short: {video_path} ({total_frames} frames, {duration:.1f}s)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                # Check frame quality
                if frame is not None and frame.size > 0:
                    try:
                        # Convert color space (handle different formats)
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            # Apply CLAHE for better contrast
                            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            l = clahe.apply(l)
                            frame_processed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
                        else:
                            # Handle grayscale or other formats
                            if len(frame.shape) == 2:
                                frame_processed = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                            else:
                                frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Resize and normalize
                        resized = cv2.resize(frame_processed, (img_size, img_size))
                        normalized = resized.astype('float32') / 255.0
                        frames.append(normalized)
                        successful_frames += 1
                        
                    except Exception as e:
                        print(f"Warning: Error processing frame {frame_count} in {video_path}: {e}")
                        continue
                    
            frame_count += 1
            
            # Safety check to prevent infinite loops (especially for FLV)
            if frame_count > 100000:  # Arbitrary large number
                print(f"Warning: Processing stopped at frame {frame_count} for {video_path}")
                break
    
    except Exception as e:
        print(f"Error during frame extraction from {video_path}: {e}")
        return None
    
    finally:
        cap.release()
    
    # Validate extracted frames
    if len(frames) < MIN_FRAMES:
        print(f"Warning: Too few frames extracted from {video_path} ({len(frames)} frames)")
        return None
    
    print(f"  Extracted {len(frames)} frames from {os.path.basename(video_path)} ({file_ext})")
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
        
        # Random noise addition (subtle)
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.02, aug_frame.shape)
            aug_frame = np.clip(aug_frame + noise, 0, 1)
            
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
        
        try:
            batch_embeddings = cnn_model.predict(batch_array, verbose=0)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error in batch processing: {e}")
            continue
    
    return np.array(embeddings)

# Enhanced dataset preparation with better error handling
def prepare_dataset(data_dir, cnn_model, frame_skip=10, img_size=224, seq_len=30, 
                   augment=False, augment_prob=0.5):
    """Prepare dataset with optional augmentation and multi-format support"""
    X_sequences = []
    y_labels = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory does not exist: {data_dir}")
        return None, None
    
    emotions = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Statistics
    valid_videos = 0
    skipped_videos = 0
    format_stats = {}
    
    print(f"\nProcessing {data_dir}")
    print(f"Emotions found: {emotions}")
    print(f"Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}")
    
    for emotion in emotions:
        emotion_folder = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        
        # Get all supported video files
        video_files = get_video_files(emotion_folder)
        
        if not video_files:
            print(f"Warning: No supported video files found in {emotion_folder}")
            continue
        
        # Count formats
        for video_file in video_files:
            ext = os.path.splitext(video_file)[1].lower()
            format_stats[ext] = format_stats.get(ext, 0) + 1
        
        print(f"\nProcessing {emotion}: {len(video_files)} videos")
        if len(video_files) > 0:
            formats_in_emotion = {}
            for vf in video_files:
                ext = os.path.splitext(vf)[1].lower()
                formats_in_emotion[ext] = formats_in_emotion.get(ext, 0) + 1
            print(f"  Formats: {dict(formats_in_emotion)}")
        
        # Progress bar for each emotion
        for fname in tqdm(video_files, desc=f"  {emotion}"):
            video_path = os.path.join(emotion_folder, fname)
            
            try:
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
                    
                    # For training, add augmented version with lower probability
                    if augment and 'train' in data_dir.lower() and np.random.random() > 0.7:
                        aug_frames = augment_frames(frames, augment_prob=0.8)
                        aug_embeddings = frames_to_embeddings(aug_frames, cnn_model)
                        if len(aug_embeddings) > 0:
                            X_sequences.append(aug_embeddings)
                            y_labels.append(emotion)
                            valid_videos += 1
                
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                skipped_videos += 1
                continue
    
    print(f"\nDataset Summary for {data_dir}:")
    print(f"  Valid videos: {valid_videos}")
    print(f"  Skipped videos: {skipped_videos}")
    print(f"  Total sequences: {len(X_sequences)}")
    print(f"  Format distribution: {format_stats}")
    
    if len(X_sequences) == 0:
        print(f"ERROR: No valid sequences found in {data_dir}")
        return None, None
    
    # Pad sequences
    try:
        X_padded = pad_sequences(
            X_sequences,
            maxlen=seq_len,
            dtype='float32',
            padding='post',
            truncating='post'
        )
        print(f"  Padded sequence shape: {X_padded.shape}")
        
    except Exception as e:
        print(f"Error during sequence padding: {e}")
        return None, None
    
    return X_padded, y_labels

def print_opencv_info():
    """Print OpenCV build information for debugging"""
    print("\nOpenCV Information:")
    print(f"  Version: {cv2.__version__}")
    try:
        build_info = cv2.getBuildInformation()
        # Look for codec information in build info
        if "Video I/O" in build_info:
            print("  Video I/O support detected")
        if "FFMPEG" in build_info:
            print("  FFmpeg support detected")
    except:
        print("  Could not retrieve build information")

if __name__ == "__main__":
    print("="*60)
    print("Enhanced Multi-Format Video Emotion Embedding Generator")
    print("="*60)
    
    # Print system information
    print_opencv_info()
    
    print(f"\nSupported video formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}")
    
    print("\nLoading CNN model...")
    try:
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        print(f"✅ Model loaded: {CNN_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)
    
    # Validate directories
    missing_dirs = [split for split, path in DATA_SPLIT_DIRS.items() if not os.path.exists(path)]
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        print("Please ensure all data directories exist.")
        exit(1)
    
    # Process each split with augmentation for training
    datasets = {}
    
    print("\n" + "="*60)
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
    datasets['train'] = (X_train, y_train)
    
    print("\n" + "="*60)
    print("Processing Validation Data...")
    X_val, y_val = prepare_dataset(
        DATA_SPLIT_DIRS["val"], 
        cnn_model, 
        FRAME_SKIP, 
        IMG_SIZE, 
        SEQ_LEN,
        augment=False  # No augmentation for validation
    )
    datasets['val'] = (X_val, y_val)
    
    print("\n" + "="*60)
    print("Processing Test Data...")
    X_test, y_test = prepare_dataset(
        DATA_SPLIT_DIRS["test"], 
        cnn_model, 
        FRAME_SKIP, 
        IMG_SIZE, 
        SEQ_LEN,
        augment=False  # No augmentation for test
    )
    datasets['test'] = (X_test, y_test)
    
    # Check if all datasets are valid
    failed_datasets = [name for name, (X, y) in datasets.items() if X is None]
    if failed_datasets:
        print(f"\n❌ ERROR: Failed to process datasets: {failed_datasets}")
        
        # Check if we can proceed with partial data
        valid_datasets = [name for name, (X, y) in datasets.items() if X is not None]
        if not valid_datasets:
            print("No valid datasets found. Exiting.")
            exit(1)
        else:
            print(f"⚠️  Continuing with valid datasets: {valid_datasets}")
    
    # Encode labels consistently
    print("\n" + "="*60)
    print("Encoding labels...")
    
    # Collect all labels from valid datasets
    all_labels = []
    for name, (X, y) in datasets.items():
        if X is not None and y is not None:
            all_labels.extend(y)
    
    if not all_labels:
        print("❌ ERROR: No labels found in any dataset.")
        exit(1)
    
    le = LabelEncoder()
    le.fit(all_labels)
    
    # Encode labels for each dataset
    encoded_datasets = {}
    for name, (X, y) in datasets.items():
        if X is not None and y is not None:
            y_encoded = le.transform(y)
            encoded_datasets[name] = (X, y_encoded)
        else:
            encoded_datasets[name] = (None, None)
    
    print(f"Label classes: {list(le.classes_)}")
    
    # Print final statistics
    print("\n" + "="*60)
    print("FINAL DATASET STATISTICS")
    print("="*60)
    
    total_sequences = 0
    for name, (X, y) in encoded_datasets.items():
        if X is not None:
            print(f"{name.capitalize():>10}: {X.shape[0]:>5} sequences, shape: {X.shape}")
            total_sequences += X.shape[0]
        else:
            print(f"{name.capitalize():>10}: {'N/A':>5} (failed to process)")
    
    print(f"{'Total':>10}: {total_sequences:>5} sequences")
    
    # Print class distribution
    print(f"\nClass Distribution:")
    print(f"{'Emotion':<12} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-" * 40)
    
    for emotion in le.classes_:
        emotion_counts = []
        for name in ['train', 'val', 'test']:
            X, y = encoded_datasets[name]
            if y is not None:
                count = np.sum(y == le.transform([emotion])[0])
                emotion_counts.append(f"{count:>5}")
            else:
                emotion_counts.append("N/A")
        
        print(f"{emotion:<12} {emotion_counts[0]:<8} {emotion_counts[1]:<8} {emotion_counts[2]:<8}")
    
    # Save with additional metadata
    output_file = "embeddings_multiformat_crema_gru.npz"
    
    # Prepare data for saving
    save_data = {
        'label_classes': le.classes_,
        'frame_skip': FRAME_SKIP,
        'seq_len': SEQ_LEN,
        'img_size': IMG_SIZE,
        'supported_formats': SUPPORTED_VIDEO_FORMATS
    }
    
    # Add valid datasets
    for name, (X, y) in encoded_datasets.items():
        if X is not None and y is not None:
            save_data[f'X_{name}'] = X
            save_data[f'y_{name}'] = y
    
    try:
        np.savez(output_file, **save_data)
        print(f"\n✅ Saved balanced multi-format embeddings to '{output_file}'")
        print(f"📁 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        print("⚖️  Training data is class-balanced for equal emotion representation")
        print("🚀 Ready for GRU training with balanced dataset!")
        
    except Exception as e:
        print(f"❌ Error saving embeddings: {e}")
        exit(1)
    
    print("\n" + "="*60)
    print("Processing completed successfully!")
    print("="*60)
