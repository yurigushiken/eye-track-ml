#!/usr/bin/env python3
# 1_preprocessing.py

import os
import glob
import subprocess
import time
import re
from datetime import datetime, timedelta

# Input and output directories
VIDEO_DIR = r"D:\infant eye-tracking\paper-area\0_participant_videos"
OUTPUT_DIR = r"D:\infant eye-tracking\paper-area\1_preprocessing_output"

def get_frame_number(line):
    """Extract frame number from ffmpeg output line"""
    match = re.search(r'frame=\s*(\d+)', line)
    if match:
        return int(match.group(1))
    return None

def format_time(seconds):
    """Format seconds into MM:SS"""
    return str(timedelta(seconds=int(seconds))).split('.')[0]

def update_progress_display(video_name, frame_num, total_frames, start_time, current_frame_file):
    """Update the 12-line progress display"""
    elapsed = time.time() - start_time
    if frame_num > 0:
        frames_per_second = frame_num / elapsed
        remaining_frames = total_frames - frame_num
        eta_seconds = remaining_frames / frames_per_second if frames_per_second > 0 else 0
    else:
        eta_seconds = 0
    
    progress = (frame_num / total_frames) if total_frames > 0 else 0
    
    # Prepare 12 lines of status
    lines = [
        f"Processing: {video_name}",
        f"Frame: {frame_num}/{total_frames}",
        f"Progress: {progress:.1%}",
        f"Writing: {current_frame_file}",
        f"Elapsed: {format_time(elapsed)}",
        f"ETA: {format_time(eta_seconds)}",
        f"Frames/sec: {frame_num/elapsed:.1f}" if elapsed > 0 else "Frames/sec: --",
        "",
        f"Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}",
        f"Est. Finish: {datetime.fromtimestamp(start_time + eta_seconds).strftime('%H:%M:%S')}",
        "",
        "Press Ctrl+C to cancel"
    ]
    
    # Clear previous lines and print new status
    print("\033[F" * 12 + "\033[K" + "\n".join(lines))

def create_1024_frames(video_path):
    """
    Process video frames to 1024x1024 format if not already done
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(OUTPUT_DIR, f"{video_name}-1024-frames")
    
    if os.path.isdir(output_dir):
        print(f"Skipping (already processed): {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    # Get total frames using ffprobe
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        video_path
    ]
    total_frames = int(subprocess.check_output(probe_cmd).decode().strip())

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "pad=1024:1024:0:128:black",
        "-progress", "pipe:1",
        os.path.join(output_dir, "frame_%05d.png")
    ]

    # Initialize progress display
    print("\n" * 12)  # Make space for progress display
    start_time = time.time()
    current_frame = 0
    current_frame_file = ""

    # Run ffmpeg with progress monitoring
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    while True:
        # Read output line by line
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        
        if output:
            # Update current frame number
            frame_num = get_frame_number(output)
            if frame_num is not None:
                current_frame = frame_num
                current_frame_file = f"frame_{current_frame:05d}.png"
            
            # Update progress display
            update_progress_display(
                video_name,
                current_frame,
                total_frames,
                start_time,
                current_frame_file
            )

    # Show final progress
    update_progress_display(
        video_name,
        total_frames,
        total_frames,
        start_time,
        "Complete"
    )
    print("\n")  # Add a blank line after completion

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.avi"))
    
    print(f"Found {len(video_files)} videos to process")
    time.sleep(1)
    
    for video_file in video_files:
        create_1024_frames(video_file)

if __name__ == "__main__":
    main()