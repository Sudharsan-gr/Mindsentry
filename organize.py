import os
import shutil

# Path to the folder where your RAVDESS videos are stored
SOURCE_DIR = "C:/Users/grsud/Desktop/MIND SENTRY/Data/Video"  
# Path where emotion folders will be created and videos moved
TARGET_DIR = "C:/Users/grsud/Desktop/MIND SENTRY/Data/GRU training"        

# Emotion code to emotion name map (RAVDESS)
emotion_dict = {
    '01': 'neutral',
    '02': 'calm',      # You can delete this folder later if you want
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def get_emotion_from_filename(filename):
    parts = filename.split('-')
    if len(parts) > 2:
        emotion_code = parts[2]
        return emotion_dict.get(emotion_code, "unknown")
    return "unknown"

def organize_videos_by_emotion(source_dir, target_dir):
    files = [f for f in os.listdir(source_dir) if f.endswith(".mp4")]
    for f in files:
        emotion = get_emotion_from_filename(f)
        if emotion == "unknown":
            print(f"Skipping unknown emotion file: {f}")
            continue

        emotion_folder = os.path.join(target_dir, emotion)
        os.makedirs(emotion_folder, exist_ok=True)

        src_path = os.path.join(source_dir, f)
        dst_path = os.path.join(emotion_folder, f)

        print(f"Moving '{f}' to folder '{emotion}'")
        shutil.move(src_path, dst_path)

if __name__ == "__main__":
    organize_videos_by_emotion(SOURCE_DIR, TARGET_DIR)
    print("Video files reorganized by emotion folders.")
