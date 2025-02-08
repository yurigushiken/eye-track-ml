#!/usr/bin/env python3
# 3_csv_objects_yolo.py

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re

# Import configuration variables for step 3
from config import CSV_YOLO_INPUT_DIR, CSV_YOLO_OUTPUT_DIR

def get_bbox(detection):
    """
    Returns (x_min, y_min, x_max, y_max) from the YOLO JSON detection,
    which has x_center, y_center, width, height.
    """
    x_center = detection.get('x_center', 0)
    y_center = detection.get('y_center', 0)
    w = detection.get('width', 0)
    h = detection.get('height', 0)
    x_min = x_center - w / 2
    x_max = x_center + w / 2
    y_min = y_center - h / 2
    y_max = y_center + h / 2
    return x_min, y_min, x_max, y_max

def check_bbox_intersection(bbox1, bbox2):
    """
    Checks whether two bounding boxes intersect or even touch.
    bbox format: (x_min, y_min, x_max, y_max).
    """
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    if x_max1 < x_min2 or x_min1 > x_max2 or y_max1 < y_min2 or y_min1 > y_max2:
        return False
    return True

def normalize_class_name(class_name):
    """
    Normalizes class names: strip whitespace, make lowercase, replace spaces with underscores.
    """
    if not isinstance(class_name, str):
        return ""
    return class_name.strip().lower().replace(' ', '_')

def extract_frame_number(filename):
    """
    Extracts frame number from the filename (assuming the second underscore-separated part
    is the frame number, e.g. 'img_00010_xxx.json' -> frame_number=10).
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None

def calculate_time(frame_number, fps=30):
    """
    Converts frame number to a time string HH:MM:SS:MMMM (4 digits for the last part)
    and also returns total_seconds.
    """
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    remainder = total_seconds % 3600
    minutes = int(remainder // 60)
    seconds = int(remainder % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 10000))
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:04d}"
    return total_seconds, time_str

def determine_what_where(detections, priority_order):
    """
    Determines 'What' and 'Where' by checking bounding-box overlap between the 'blue_dot'
    and each class in a defined priority order.
    """
    if not detections or not isinstance(detections, list):
        return ('no', 'signal')
    try:
        blue_dots = [d for d in detections if normalize_class_name(d.get('class', '')) == 'blue_dot']
        if not blue_dots:
            return ('no', 'signal')
        blue_dot = max(blue_dots, key=lambda x: x.get('confidence', 0))
        bd_box = get_bbox(blue_dot)
        class_mapping = {
            'toy': ('toy', 'other'),
            'toy2': ('toy2', 'other'),
            'hand_man': ('man', 'hands'),
            'hand_woman': ('woman', 'hands'),
            'face_woman': ('woman', 'face'),
            'face_man': ('man', 'face'),
            'body_woman': ('woman', 'body'),
            'body_man': ('man', 'body'),
            'green_circle': ('green_circle', 'other'),
            'screen': ('screen', 'other')
        }
        for obj_class in priority_order:
            matching_objs = [d for d in detections if normalize_class_name(d.get('class', '')) == normalize_class_name(obj_class)]
            matching_objs.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            for obj in matching_objs:
                obj_box = get_bbox(obj)
                if check_bbox_intersection(bd_box, obj_box):
                    key = normalize_class_name(obj.get('class', ''))
                    return class_mapping.get(key, (obj['class'].replace('_', ' '), 'other'))
        return ('screen', 'other')
    except Exception as e:
        print(f"Error in determine_what_where: {e}")
        return ('unknown', 'unknown')

def get_blue_dot_center(detections):
    """
    Returns the center (x_center, y_center) of the highest confidence 'blue_dot' as a string.
    If none found, returns ''.
    Added debugging print statements.
    """
    print("Entering get_blue_dot_center function...")  # Debug print

    try:
        blue_dots = [d for d in detections if normalize_class_name(d.get('class', '')) == 'blue_dot']
        print(f"Number of blue_dot detections found: {len(blue_dots)}") # Debug print

        if not blue_dots:
            print("No blue_dot detections found.") # Debug print
            return ''

        blue_dot = max(blue_dots, key=lambda x: x.get('confidence', 0))
        x_c = blue_dot.get('x_center', 0)
        y_c = blue_dot.get('y_center', 0)
        center_str = f"({x_c:.2f}, {y_c:.2f})"
        print(f"Blue dot center calculated: {center_str}") # Debug print
        return center_str

    except KeyError as e:
        print(f"KeyError in get_blue_dot_center: {e}") # Debug print - specific error
        return ''
    except TypeError as e:
        print(f"TypeError in get_blue_dot_center: {e}") # Debug print - specific error
        print(f"Detections data causing TypeError: {detections}") # Print the data
        return ''
    except Exception as e:
        print(f"General error in get_blue_dot_center: {e}") # Debug print - general error
        return ''
    finally:
        print("Exiting get_blue_dot_center function.\n") # Debug print


def parse_participant_id(dir_name):
    """
    Removes any parentheses and their contents from the directory name,
    then strips extra whitespace.
    """
    import re
    name_without_parens = re.sub(r'\(.*?\)', '', dir_name)
    return name_without_parens.strip()

def create_datasheet_filename(participant_folder_name):
    """
    Creates a datasheet filename by taking the first three hyphen-separated parts
    of the participant folder name and appending '-datasheet.csv'.
    For example, 'FiftySix-0501-1673-1024-segmentation' becomes 'FiftySix-0501-1673-datasheet.csv'.
    """
    parts = participant_folder_name.split("-")
    if len(parts) >= 3:
        base = "-".join(parts[:3])
    else:
        base = participant_folder_name
    return f"{base}-datasheet.csv"

def process_detections(detections_dir, output_folder, participant_id, fps=30):
    """
    Process all detection files in a directory and create a CSV datasheet.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    filename = create_datasheet_filename(participant_id)
    output_csv = output_folder / filename
    if output_csv.exists():
        print(f"Datasheet already exists for {participant_id} at {output_csv}, skipping processing.")
        return True
    data = []
    priority_order = ['toy', 'toy2', 'hand_man', 'hand_woman', 'face_woman', 'face_man', 'body_woman', 'body_man', 'green_circle', 'screen']
    json_files = sorted(detections_dir.glob("*.json"), key=lambda x: extract_frame_number(x.name) if extract_frame_number(x.name) is not None else -1)
    if not json_files:
        print(f"No JSON files found in {detections_dir}")
        return False
    for json_file in tqdm(json_files, desc=f"Processing frames for {participant_id}"):
        frame_number = extract_frame_number(json_file.name)
        if frame_number is None:
            print(f"Warning: Invalid frame number in {json_file.name}")
            continue
        total_seconds, time_str = calculate_time(frame_number, fps=fps)
        try:
            with open(json_file, 'r') as f:
                detections = json.load(f)
            if not isinstance(detections, list):
                print(f"Warning: Invalid detections format in {json_file.name}")
                continue
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            continue
        what, where = determine_what_where(detections, priority_order)
        blue_dot_center = get_blue_dot_center(detections)

        print(f"Debug: Blue Dot Center value before appending to data: {blue_dot_center}") # <--- ADDED DEBUG PRINT

        data.append({
            "Participant": participant_id,
            "Frame Number": frame_number,
            "Time": time_str,
            "What": what,
            "Where": where,
            "Onset": f"{total_seconds:.4f}",
            "Offset": f"{(total_seconds + 1/fps):.4f}",
            "Blue Dot Center": blue_dot_center
        })
    if data:
        df = pd.DataFrame(data, columns=["Participant", "Frame Number", "Time", "What", "Where", "Onset", "Offset", "Blue Dot Center"])
        df.to_csv(output_csv, index=False)
        print(f"Data sheet saved to: {output_csv}")
        return True
    return False

def find_and_process_all_detections(base_dir):
    """
    Recursively find and process all 'detections' directories under the base directory.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"Base directory does not exist: {base_dir}")
        return
    detection_dirs = [d for d in base_dir.rglob("*") if d.is_dir() and d.name.startswith("detections")]
    if not detection_dirs:
        print(f"No 'detections' directories found under {base_dir}")
        return
    print(f"Found {len(detection_dirs)} 'detections' directories to process.")
    global_datasheet_folder = Path(CSV_YOLO_OUTPUT_DIR) # <---- CONVERT TO PATH OBJECT HERE
    global_datasheet_folder.mkdir(parents=True, exist_ok=True)
    processed_count = 0
    failed_dirs = []
    for detections_dir in detection_dirs:
        try:
            print(f"\nProcessing directory: {detections_dir}")
            participant_folder = detections_dir.parent
            participant_id = parse_participant_id(participant_folder.name)
            print(f"Participant ID (processed): {participant_id}")
            output_folder = global_datasheet_folder
            if process_detections(detections_dir, output_folder, participant_id):
                processed_count += 1
            else:
                failed_dirs.append(str(detections_dir))
        except Exception as e:
            print(f"Error processing {detections_dir}: {e}")
            failed_dirs.append(str(detections_dir))
    print("\nProcessing Summary:")
    print(f"Total directories found: {len(detection_dirs)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed to process: {len(failed_dirs)}")
    if failed_dirs:
        print("\nFailed directories:")
        for dir_path in failed_dirs:
            print(f"- {dir_path}")

def main():
    base_dir = CSV_YOLO_INPUT_DIR
    print("Starting detection processing...")
    find_and_process_all_detections(base_dir)
    print("\nAll processing completed.")

if __name__ == "__main__":
    main()