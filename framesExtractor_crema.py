import os
import cv2

# Root folder containing 'train', 'val', 'test'
CREMA_ROOT = r"C:\Users\grsud\Desktop\MIND SENTRY\Data\CREMA"

# Output root folder for extracted frames
OUTPUT_ROOT = r"C:\Users\grsud\Desktop\MIND SENTRY\Data\CREMA_frames"

FRAME_SKIP = 10  # extract every 10th frame

def extract_frames_from_crema():
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(CREMA_ROOT, split)
        for emotion in os.listdir(split_dir):
            emotion_dir = os.path.join(split_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            
            output_emotion_dir = os.path.join(OUTPUT_ROOT, split, emotion)
            os.makedirs(output_emotion_dir, exist_ok=True)
            
            for video_file in os.listdir(emotion_dir):
                if not video_file.lower().endswith(('.mp4', '.flv', '.avi')):
                    continue
                video_path = os.path.join(emotion_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                
                frame_count = 0
                saved_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % FRAME_SKIP == 0:
                        frame_filename = f"{os.path.splitext(video_file)[0]}_frame{saved_count}.jpg"
                        save_path = os.path.join(output_emotion_dir, frame_filename)
                        cv2.imwrite(save_path, frame)
                        saved_count += 1
                    frame_count += 1
                
                cap.release()
                print(f"Extracted {saved_count} frames from {video_file} in {split}/{emotion}")

if __name__ == "__main__":
    extract_frames_from_crema()
