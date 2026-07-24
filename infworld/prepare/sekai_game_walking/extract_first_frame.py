#!/usr/bin/env python3
"""Extract first frame from video.mp4 in each case folder and save as image.jpg"""

import os
from pathlib import Path
import cv2


def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video file and save as JPEG"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read first frame from {video_path}")
        return False

    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Extracted: {output_path}")
    return True


def main():
    dataset_root = Path("/mnt/efs/chenran/ttt/infworld/dataset/sekai-game-walking-854_480_30fps")

    # Get all case folders and sort them numerically
    case_folders = sorted(
        [d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("case")],
        key=lambda x: int(x.name.replace("case", ""))
    )

    # Process all cases
    print(f"Found {len(case_folders)} case folders. Processing all...")

    success_count = 0
    for case_folder in case_folders:
        video_path = case_folder / "video.mp4"
        output_path = case_folder / "image.jpg"

        if not video_path.exists():
            print(f"Warning: {video_path} does not exist, skipping")
            continue

        if extract_first_frame(video_path, output_path):
            success_count += 1

    print(f"\nCompleted: {success_count}/{len(case_folders)} frames extracted successfully")


if __name__ == "__main__":
    main()
