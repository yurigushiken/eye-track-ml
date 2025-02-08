# 3_csv_objects_yolo_sam.py
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import configuration variables for step 3 (YOLO_SAM version)
from config import CSV_YOLO_SAM_INPUT_DIR, CSV_YOLO_SAM_OUTPUT_DIR

def create_new_output_dir(base_path, prefix="datasheet"):
    """Creates a new directory with an incremented suffix"""
    base_path = Path(base_path)
    suffix = 1
    while True:
        new_dir = base_path / f"{prefix}-{suffix:02d}"
        if not new_dir.exists():
            new_dir.mkdir(parents=True, exist_ok=False)
            return new_dir
        suffix += 1

def extract_frame_number(filename):
    """Extracts frame number from filename"""
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None

def calculate_time(frame_number, fps=30):
    """Converts frame number to time string"""
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    remainder = total_seconds % 3600
    minutes = int(remainder // 60)
    seconds = int(remainder % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
    return total_seconds, time_str

def create_mask_array(mask_pixels):
    """
    Creates a boolean mask array from mask pixels.
    """
    if not mask_pixels or len(mask_pixels) < 3:
        return None

    try:
        points = np.array(mask_pixels)

        # Get image dimensions from max coordinates
        height = int(np.max(points[:, 0])) + 1
        width = int(np.max(points[:, 1])) + 1

        # Create empty mask
        mask = np.zeros((height, width), dtype=bool)

        # Fill mask using points
        # Convert to integer indices
        y_coords = points[:, 0].astype(int)
        x_coords = points[:, 1].astype(int)

        # Set mask pixels to True
        mask[y_coords, x_coords] = True

        return mask
    except Exception as e:
        print(f"Error creating mask array: {e}")
        return None

def check_mask_intersection(mask1_pixels, mask2_pixels):
    """
    Check if two masks intersect by checking actual pixels.
    Returns True if there is any overlap between the masks.
    """
    if not mask1_pixels or not mask2_pixels:
        return False

    try:
        # Create mask arrays
        mask1 = create_mask_array(mask1_pixels)
        mask2 = create_mask_array(mask2_pixels)

        if mask1 is None or mask2 is None:
            return False

        # Make sure masks are the same size
        max_height = max(mask1.shape[0], mask2.shape[0])
        max_width = max(mask1.shape[1], mask2.shape[1])

        # Resize masks if needed
        if mask1.shape != (max_height, max_width):
            new_mask1 = np.zeros((max_height, max_width), dtype=bool)
            new_mask1[:mask1.shape[0], :mask1.shape[1]] = mask1
            mask1 = new_mask1

        if mask2.shape != (max_height, max_width):
            new_mask2 = np.zeros((max_height, max_width), dtype=bool)
            new_mask2[:mask2.shape[0], :mask2.shape[1]] = mask2
            mask2 = new_mask2

        # Check for any intersection
        intersection = np.logical_and(mask1, mask2)
        return np.any(intersection)

    except Exception as e:
        print(f"Error checking mask intersection: {e}")
        return False

def normalize_class_name(class_name):
    """Normalizes class names"""
    if not isinstance(class_name, str):
        return ""
    return class_name.strip().lower().replace(' ', '_')

def determine_what_where(detections, priority_order):
    """
    Determines What and Where based on actual mask intersections,
    following the strict priority order.
    Expects detections to be a list of dictionaries, where each dictionary
    contains BOTH bounding box AND mask_pixels for each object.
    """
    print("\n--- determine_what_where ---") # DEBUG
    print(f"Detections: {detections}") # DEBUG
    if not detections or not isinstance(detections, list):
        print("No detections or not a list - returning no signal") # DEBUG
        return ('no', 'signal')

    try:
        # Find blue dot
        blue_dots = [d for d in detections if normalize_class_name(d.get('class', '')) == 'blue_dot']
        print(f"Blue dots found: {blue_dots}") # DEBUG
        if not blue_dots:
            print("No blue dots found - returning no signal") # DEBUG
            return ('no', 'signal')

        # Use most confident blue dot
        blue_dot = max(blue_dots, key=lambda x: x.get('confidence', 0))
        print(f"Most confident blue dot: {blue_dot}") # DEBUG
        blue_dot_mask = blue_dot.get('mask_pixels', [])
        print(f"Blue dot mask pixels: {blue_dot_mask}") # DEBUG

        if not blue_dot_mask:
            print("Blue dot has no mask - returning no signal") # DEBUG
            return ('no', 'signal')

        # Class mapping defines what labels to output for each detected class
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

        # Check intersections following priority order
        for obj_class in priority_order:
            print(f"\nChecking priority class: {obj_class}") # DEBUG
            matching_objs = [d for d in detections
                           if normalize_class_name(d.get('class', '')) == normalize_class_name(obj_class)]
            print(f"Matching objects for class {obj_class}: {matching_objs}") # DEBUG

            # Sort by confidence if multiple objects of same class
            matching_objs.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            for obj in matching_objs:
                obj_mask = obj.get('mask_pixels', [])
                print(f"Object: {obj}, Mask pixels: {obj_mask}") # DEBUG
                if not obj_mask:
                    print(f"Object {obj['class']} has no mask - skipping intersection check") # DEBUG
                    continue

                # Check actual mask intersection
                intersection_result = check_mask_intersection(blue_dot_mask, obj_mask)
                print(f"Intersection check between blue_dot and {obj['class']}: {intersection_result}") # DEBUG
                if intersection_result:
                    key = normalize_class_name(obj.get('class', ''))
                    what_where = class_mapping.get(key, (obj['class'].replace('_', ' '), 'other'))
                    print(f"Intersection found with {obj['class']} - returning What, Where: {what_where}") # DEBUG
                    return what_where

        # If no intersections found
        print("No intersections found in priority classes - returning screen, other") # DEBUG
        return ('screen', 'other')

    except Exception as e:
        print(f"Error in determine_what_where: {e}") # DEBUG
        return ('unknown', 'unknown')

def process_detections(detections_dir, output_base, participant_id, fps=30):
    """Process all detection files in directory"""
    # Create output directory
    output_dir = Path(CSV_YOLO_SAM_OUTPUT_DIR) # Use configured output dir, don't create new one each time.
    output_dir.mkdir(parents=True, exist_ok=True) # ensure it exists
    output_csv_filename = f"{participant_id}-datasheet.csv" # create filename based on participant ID
    output_csv = output_dir / output_csv_filename
    print(f"Output CSV will be saved at: {output_csv}")

    # Initialize data list
    data = []
    priority_order = [
        'toy',
        'toy2',
        'hand_man',
        'hand_woman',
        'face_woman',
        'face_man',
        'body_woman',
        'body_man',
        'green_circle',
        'screen'
    ]

    # Get JSON files
    json_files = sorted(
        detections_dir.glob("*.json"),
        key=lambda x: extract_frame_number(x.name) if extract_frame_number(x.name) is not None else -1
    )

    if not json_files:
        print(f"No JSON files found in {detections_dir}")
        return

    # Process each JSON file
    for json_file in tqdm(json_files, desc=f"Processing frames in {detections_dir}"):
        frame_number = extract_frame_number(json_file.name)
        if frame_number is None:
            print(f"Warning: Invalid frame number in {json_file.name}")
            continue

        total_seconds, time_str = calculate_time(frame_number, fps=fps)

        try:
            with open(json_file, 'r') as f:
                raw_detections = json.load(f)
            if not isinstance(raw_detections, list):
                print(f"Warning: Invalid detections format in {json_file.name}")
                continue

            # --- Restructure detections to merge bbox and mask info ---
            merged_detections = {}
            for det in raw_detections:
                key = (det.get('class'), det.get('confidence')) # Unique identifier for each detection
                if key not in merged_detections:
                    merged_detections[key] = {'class': det.get('class'), 'confidence': det.get('confidence')} # Initialize
                if 'x_center' in det: # Bounding box info
                    merged_detections[key].update({
                        'x_center': det.get('x_center'),
                        'y_center': det.get('y_center'),
                        'width': det.get('width'),
                        'height': det.get('height')
                    })
                if 'mask_pixels' in det: # Mask pixels info
                    merged_detections[key]['mask_pixels'] = det.get('mask_pixels')

            detections_list = list(merged_detections.values()) # Convert back to list for processing
            # --- End of restructuring ---

        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            continue

        what, where = determine_what_where(detections_list, priority_order) # Use the restructured detections

        data.append({
            "Participant": participant_id,
            "Frame Number": frame_number,
            "Time": time_str,
            "What": what,
            "Where": where,
            "Onset": f"{total_seconds:.2f}",
            "Offset": f"{(total_seconds + 1/fps):.2f}"
        })

    # Create DataFrame and save CSV
    df = pd.DataFrame(data, columns=[
        "Participant",
        "Frame Number",
        "Time",
        "What",
        "Where",
        "Onset",
        "Offset"
    ])

    df.to_csv(output_csv, index=False)
    print(f"Data sheet saved to: {output_csv}")

def find_and_process_participant_detections(base_dir):
    """
    Recursively find and process 'detections' directories for each participant under the base directory.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"Base directory does not exist: {base_dir}")
        return

    participant_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not participant_dirs:
        print(f"No participant directories found under {base_dir}")
        return

    print(f"Found {len(participant_dirs)} participant directories to process.")

    for participant_folder in participant_dirs:
        detections_dir = participant_folder / "inference_output" / "detections" # Assuming "inference_output/detections" is consistent
        if detections_dir.exists() and detections_dir.is_dir():
            try:
                participant_id = participant_folder.name # Use folder name as participant ID
                print(f"\nProcessing participant: {participant_id}, detections dir: {detections_dir}")
                process_detections(detections_dir, CSV_YOLO_SAM_OUTPUT_DIR, participant_id) # Use configured output path
            except Exception as e:
                print(f"Error processing participant {participant_id}: {e}")
        else:
            print(f"Warning: No detections directory found for participant: {participant_folder.name} at {detections_dir}")


def main():
    """Main function"""
    detections_base_dir = CSV_YOLO_SAM_INPUT_DIR # Use configured input dir
    print("Starting detection processing...")
    find_and_process_participant_detections(detections_base_dir) # Use function to find and process all participant detections
    print("\nProcessing completed.")


if __name__ == "__main__":
    main()