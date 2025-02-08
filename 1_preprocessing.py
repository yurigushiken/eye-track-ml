#!/usr/bin/env python3
# 1_preprocessing.py

import os
import glob
import subprocess
import time

from config import VIDEO_DIR, PREPROCESSING_OUTPUT_DIR

def create_1024_frames(video_path):
    """Process video frames to 1024x1024 format"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(PREPROCESSING_OUTPUT_DIR, f"{video_name}-1024-frames")
    
    if os.path.isdir(output_dir):
        print(f"Skipping (already processed): {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing: {video_name}")
    print("This may take a few minutes...\n")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "pad=1024:1024:0:128:black",
        "-f", "image2",
        "-y",
        os.path.join(output_dir, "frame_%05d.png")
    ]

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if process.returncode != 0:
            print(f"Error processing video: {process.stderr}")
            return
            
        print(f"Completed processing: {video_name}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return

def main():
    os.makedirs(PREPROCESSING_OUTPUT_DIR, exist_ok=True)
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.avi"))
    
    print(f"Found {len(video_files)} videos to process")
    
    for video_file in video_files:
        create_1024_frames(video_file)
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()