import os
import shutil
import random

src_root = r"C:\Users\grsud\Desktop\MIND SENTRY\Data\GRU training"
output_root = r"C:\Users\grsud\Desktop\MIND SENTRY\Data\GRUEqual split"  # output base folder

# Create base output directories if not exist
for split in ['train', 'val', 'test']:
    split_path = os.path.join(output_root, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)

classes = os.listdir(src_root)

for emotion in classes:
    emotion_path = os.path.join(src_root, emotion)
    videos = os.listdir(emotion_path)
    random.shuffle(videos)

    # Define splits
    train_split = int(0.7 * len(videos))
    val_split = int(0.85 * len(videos))

    train_videos = videos[:train_split]
    val_videos = videos[train_split:val_split]
    test_videos = videos[val_split:]

    # Create emotion folders inside train/val/test
    for split_name, split_videos in zip(['train', 'val', 'test'], [train_videos, val_videos, test_videos]):
        class_output_path = os.path.join(output_root, split_name, emotion)
        os.makedirs(class_output_path, exist_ok=True)

        for video_file in split_videos:
            src_video_path = os.path.join(emotion_path, video_file)
            dst_video_path = os.path.join(class_output_path, video_file)
            shutil.copy2(src_video_path, dst_video_path)  # copy video file

print("Dataset split completed successfully!")
