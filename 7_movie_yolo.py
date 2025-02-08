#!/usr/bin/env python3
# 7_movie.py

import os
import csv
import subprocess
import re
import time
import shutil

# ============================================
# TOGGLE SETTINGS FOR TEXT OVERLAYS
# ============================================
SHOW_LEFT_OVERLAY = True   # If False, hides frame # and "What, Where"
SHOW_RIGHT_OVERLAY = True  # If False, hides "time, event_verified, trial_number_global, segment"
SHOW_BOTTOM_OVERLAY = True # If False, hides "participant type & age"

# Import configuration variables for movie creation
from config import MOVIE_FRAMES_BASE_DIR, MOVIE_DATASHEET_CSV_DIR, MOVIE_PARTICIPANT_CSV_DIR, MOVIE_OUTPUT_DIR, MOVIE_FONT_PATH

FONT_PATH_SERIF = MOVIE_FONT_PATH

def read_csv_with_fallback_encodings(filepath, fallback_encodings=None):
    if fallback_encodings is None:
        fallback_encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in fallback_encodings:
        try:
            with open(filepath, mode="r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError(f"Could not decode file {filepath} with any of the tried encodings: {fallback_encodings}")

def escape_drawtext(text):
    if not text:
        return ""
    text = text.replace(':', '\\:')
    text = text.replace("'", "\\'")
    return text

def frame_to_timecode(frame_num, fps=30):
    total_seconds = frame_num / float(fps)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

def is_hidden_or_resource_file(fname):
    return fname.startswith('.') or fname.startswith('._')

def find_datasheet_csv(datasheet_dir, participant_name):
    datasheet_pattern = re.compile(rf"^{participant_name}.*-datasheet\.csv$", re.IGNORECASE)
    for fname in os.listdir(datasheet_dir):
        if not fname.lower().endswith(".csv"):
            continue
        if is_hidden_or_resource_file(fname):
            continue
        if datasheet_pattern.match(fname):
            full_path = os.path.join(datasheet_dir, fname)
            if os.path.isfile(full_path):
                return full_path
    return None

def find_participant_csv(participant_csv_dir, participant_name):
    participant_pattern = re.compile(rf"^{participant_name}.*-maindoc\.csv$", re.IGNORECASE)
    for fname in os.listdir(participant_csv_dir):
        if not fname.lower().endswith(".csv"):
            continue
        if is_hidden_or_resource_file(fname):
            continue
        if participant_pattern.match(fname):
            full_path = os.path.join(participant_csv_dir, fname)
            if os.path.isfile(full_path):
                return full_path
    return None

def read_merged_data(participant_name):
    datasheet_csv = find_datasheet_csv(MOVIE_DATASHEET_CSV_DIR, participant_name)
    if not datasheet_csv:
        raise FileNotFoundError(f"No datasheet CSV found for participant: {participant_name} in directory: {MOVIE_DATASHEET_CSV_DIR}")
    if not os.path.exists(datasheet_csv):
        raise FileNotFoundError(f"Datasheet CSV not found at path: {datasheet_csv}")
    datasheet_rows = read_csv_with_fallback_encodings(datasheet_csv)
    if not datasheet_rows:
        raise ValueError(f"CSV '{datasheet_csv}' has no data rows.")
    if "Frame Number" not in datasheet_rows[0]:
        raise ValueError(f"CSV '{datasheet_csv}' must contain 'Frame Number' column.")
    datasheet_data = {}
    for row in datasheet_rows:
        frame_str = row.get("Frame Number", "").strip()
        if frame_str.isdigit():
            datasheet_data[int(frame_str)] = row

    participant_csv = find_participant_csv(MOVIE_PARTICIPANT_CSV_DIR, participant_name)
    participant_data = {}
    participant_type = None
    participant_age_months = None
    participant_age_years = None
    if participant_csv and os.path.exists(participant_csv):
        p_rows = read_csv_with_fallback_encodings(participant_csv)
        if p_rows:
            for p_row in p_rows:
                frame_str_p = p_row.get("Frame Number", "").strip()
                if frame_str_p.isdigit():
                    participant_data[int(frame_str_p)] = p_row
                if participant_type is None and "participant_type" in p_row:
                    participant_type = p_row["participant_type"].strip()
                if participant_age_months is None and "participant_age_months" in p_row:
                    participant_age_months = p_row["participant_age_months"].strip()
                if participant_age_years is None and "participant_age_years" in p_row:
                    participant_age_years = p_row["participant_age_years"].strip()

    csv_data_by_frame = {}
    all_frames = set(datasheet_data.keys()).union(set(participant_data.keys()))
    for fnum in sorted(all_frames):
        merged_row = {}
        if fnum in datasheet_data:
            merged_row.update(datasheet_data[fnum])
        if fnum in participant_data:
            p_row = participant_data[fnum]
            merged_row["What"] = p_row.get("What", merged_row.get("What", ""))
            merged_row["Where"] = p_row.get("Where", merged_row.get("Where", ""))
            for key, val in p_row.items():
                if key not in ["Frame Number", "What", "Where"]:
                    merged_row[key] = val
        merged_row["Frame Number"] = fnum
        csv_data_by_frame[fnum] = merged_row
    return csv_data_by_frame, participant_type, participant_age_months, participant_age_years

def create_movie_csv(participant_dir, csv_data_by_frame):
    movie_csv_path = os.path.join(participant_dir, "movie.csv")
    fieldnames = ["Participant", "Frame Number", "Time", "What", "Where", "Onset", "Offset", "Blue Dot Center", "event_verified", "frame_count_event", "trial_number_global", "frame_count_trial_number", "segment", "frame_count_segment", "participant_type", "participant_age_months", "participant_age_years"]
    sorted_frames = sorted(csv_data_by_frame.keys())
    with open(movie_csv_path, "w", newline='', encoding='utf-8') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for fnum in sorted_frames:
            row = csv_data_by_frame[fnum]
            out_row = {
                "Participant": row.get("Participant", ""),
                "Frame Number": fnum,
                "Time": row.get("Time", ""),
                "What": row.get("What", ""),
                "Where": row.get("Where", ""),
                "Onset": row.get("Onset", ""),
                "Offset": row.get("Offset", ""),
                "Blue Dot Center": row.get("Blue Dot Center", ""),
                "event_verified": row.get("event_verified", ""),
                "frame_count_event": row.get("frame_count_event", ""),
                "trial_number_global": row.get("trial_number_global", ""),
                "frame_count_trial_number": row.get("frame_count_trial_number", ""),
                "segment": row.get("segment", ""),
                "frame_count_segment": row.get("frame_count_segment", ""),
                "participant_type": row.get("participant_type", ""),
                "participant_age_months": row.get("participant_age_months", ""),
                "participant_age_years": row.get("participant_age_years", "")
            }
            writer.writerow(out_row)

def gather_frame_numbers_in_inference_dir(images_dir):
    frame_nums = []
    # This regex will capture any number of digits.
    pattern = re.compile(r"^frame_(\d+)_annotated\.jpg$", re.IGNORECASE)
    if os.path.isdir(images_dir):
        for fname in os.listdir(images_dir):
            match = pattern.match(fname)
            if match:
                frame_nums.append(int(match.group(1)))
    return sorted(frame_nums)

def process_participant_dir(participant_name, overall_start_time):
    participant_folder_name = participant_name
    # Construct images_dir path based on the example paths provided
    images_dir = os.path.join(MOVIE_FRAMES_BASE_DIR, f"{participant_folder_name}-1024-segmentation", f"image_segmentation-{participant_folder_name}-1024-segmentation")

    output_movie_dir = os.path.join(MOVIE_OUTPUT_DIR, participant_folder_name)  # output movie will be in this dir
    os.makedirs(output_movie_dir, exist_ok=True)  # create output movie dir if it doesn't exist

    avi_files_dir = os.path.dirname(MOVIE_FRAMES_BASE_DIR)  # assuming AVI files are in the parent directory of image frames, adjust if needed
    avi_files = [f for f in os.listdir(avi_files_dir) if f.lower().endswith('.avi') and participant_name in f]  # find avi file based on participant name

    if avi_files:
        avi_file_base = os.path.splitext(avi_files[0])[0]
        final_movie_name = f"{avi_file_base}-movie.mp4"
    else:
        final_movie_name = f"{participant_name}-movie.mp4"  # fallback movie name if no avi found
    mp4_output = os.path.join(output_movie_dir, final_movie_name)  # output movie path

    csv_data_by_frame, participant_type, participant_age_months, participant_age_years = read_merged_data(participant_name)  # read data using participant name
    movie_csv_path = os.path.join(output_movie_dir, "movie.csv")  # movie csv path inside participant output dir

    if not os.path.exists(movie_csv_path):
        create_movie_csv(output_movie_dir, csv_data_by_frame)
    else:
        print(f"movie.csv already exists in {output_movie_dir}. Skipping creation.")

    print(f"Searching for images in: {images_dir}")  # Debugging line: print the image directory path

    if not os.path.isdir(images_dir):
        print(f"No images folder found in: {images_dir}. Skipping.")
        return

    all_frame_nums = gather_frame_numbers_in_inference_dir(images_dir)
    if not all_frame_nums:
        print(f"No annotated frames found in {images_dir}. Skipping.")
        return

    movie_frames_dir = os.path.join(output_movie_dir, "movie_frames")  # frames with overlay will be saved here
    os.makedirs(movie_frames_dir, exist_ok=True)

    existing_overlay_pattern = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)
    existing_overlay_frames = set()
    for fname in os.listdir(movie_frames_dir):
        match = existing_overlay_pattern.match(fname)
        if match:
            existing_overlay_frames.add(int(match.group(1)))

    missing_frames = [f for f in all_frame_nums if f not in existing_overlay_frames]
    total_frames = len(all_frame_nums)
    start_time_dir = time.time()
    last_messages = []

    def print_dynamic_status(current_frame_index, total_frames):
        print("\033[H\033[J", end="")
        elapsed_dir = time.time() - start_time_dir
        elapsed_total = time.time() - overall_start_time
        if current_frame_index > 0:
            est_total_dir = (elapsed_dir / current_frame_index) * total_frames
            remain_dir = est_total_dir - elapsed_dir
        else:
            est_total_dir = 0
            remain_dir = 0
        print(f"Currently processing participant: {participant_folder_name}")
        print(f"Frame progress: {current_frame_index}/{total_frames}")
        print(f"Elapsed (this directory): {time.strftime('%H:%M:%S', time.gmtime(elapsed_dir))}")
        print(f"Estimated total (this directory): {time.strftime('%H:%M:%S', time.gmtime(est_total_dir))}")
        print(f"Estimated remaining (this directory): {time.strftime('%H:%M:%S', time.gmtime(remain_dir))}")
        print(f"Total elapsed (whole script): {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print()
        for msg in last_messages[-10:]:
            print(msg)

    if missing_frames:
        print(f"\n=== Overlay Creation Phase for {participant_folder_name} ===")
        for i, fnum in enumerate(missing_frames, start=1):
            # Update to 5-digit padding for the overlay output filename
            out_filename = f"frame_{fnum:05d}.png"
            out_path = os.path.join(movie_frames_dir, out_filename)
            if os.path.exists(out_path):
                progress_msg = f"Frame {fnum:05d} already exists. Skipping overlay."
                last_messages.append(progress_msg)
                print_dynamic_status(i, len(missing_frames))
                continue

            # Update to 5-digit padding for the input annotated image filename
            image_name = f"frame_{fnum:05d}_annotated.jpg"
            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                progress_msg = f"Missing {image_name}; skipping."
                last_messages.append(progress_msg)
                print_dynamic_status(i, len(missing_frames))
                continue

            row = csv_data_by_frame.get(fnum, {})
            filter_str = "crop=900:700:(in_w-900)/2:(in_h-700)/2"

            if SHOW_LEFT_OVERLAY:
                frame_label = f"{fnum} / {all_frame_nums[-1]}"
                what_value = row.get("What", "").strip().lower()
                where_value = row.get("Where", "").strip().lower()
                top_text = f"{what_value}, {where_value}"
                frame_label_esc = escape_drawtext(frame_label)
                top_text_esc = escape_drawtext(top_text)
                filter_str += (f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                               f"text='{frame_label_esc}':x=20:y=20:"
                               f"fontcolor=white:fontsize=24,"
                               f"drawtext=fontfile='{FONT_PATH_SERIF}':text='{top_text_esc}':"
                               f"x=(w-text_w)/2:y=40:fontcolor=white:fontsize=72")

            if SHOW_RIGHT_OVERLAY:
                time_str = frame_to_timecode(fnum, 30)
                event_verified = row.get("event_verified", "").strip()
                trial_number_global = row.get("trial_number_global", "").strip()
                segment = row.get("segment", "").strip()
                frame_count_segment = row.get("frame_count_segment", "").strip()
                event_verified_and_trial = f"{event_verified} {trial_number_global}".strip()
                time_str_esc = escape_drawtext(time_str)
                event_verified_esc = escape_drawtext(event_verified_and_trial)
                segment_esc = escape_drawtext(segment + " " + frame_count_segment)
                filter_str += (f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                               f"text='{time_str_esc}':x=(w-text_w)-20:y=20:"
                               f"fontcolor=white:fontsize=24,"
                               f"drawtext=fontfile='{FONT_PATH_SERIF}':"
                               f"text='{event_verified_esc}':"
                               f"x=(w-text_w)-20:y=50:fontcolor=white:fontsize=24,"
                               f"drawtext=fontfile='{FONT_PATH_SERIF}':"
                               f"text='{segment_esc}':"
                               f"x=(w-text_w)-20:y=80:fontcolor=white:fontsize=24")

            if SHOW_BOTTOM_OVERLAY and row.get("participant_type", ""):
                pt = row["participant_type"].lower()
                if pt == "adult" and row.get("participant_age_years", ""):
                    age_str = f"{row['participant_age_years']} years"
                elif row.get("participant_age_months", ""):
                    age_str = f"{row['participant_age_months']} months"
                else:
                    age_str = "(age unknown)"
                participant_text = f"participant: {pt} {age_str}"
                participant_text_esc = escape_drawtext(participant_text)
                filter_str += (f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                               f"text='{participant_text_esc}':"
                               f"x=(w-text_w)/2:y=(h-text_h)-60:"
                               f"fontcolor=white:fontsize=28")

            bottom_right_text = escape_drawtext("verified in part with symbolic system")
            filter_str += (f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                           f"text='{bottom_right_text}':"
                           f"x=(w-text_w)-20:y=(h-text_h)-20:"
                           f"fontcolor=white:fontsize=16")

            progress_msg = f"Processing frame {fnum:05d} -> {out_filename}"
            last_messages.append(progress_msg)
            print_dynamic_status(i, len(missing_frames))
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", image_path, "-vf", filter_str, out_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        print(f"All overlay frames already exist for {participant_folder_name}. Skipping overlay creation.")

    if os.path.exists(mp4_output):
        if not missing_frames:
            print(f"Final movie {mp4_output} already exists and no frames are missing. Skipping combine.")
            return
        else:
            print(f"Final movie {mp4_output} exists but new frames were added. Recreating the final movie.")
    else:
        print(f"\n=== Combine Phase for {participant_folder_name} ===")

    combination_msg = f"Collecting overlays from {movie_frames_dir} for final movie..."
    last_messages.append(combination_msg)
    consecutive_dir = os.path.join(movie_frames_dir, "consecutive")
    if os.path.isdir(consecutive_dir):
        shutil.rmtree(consecutive_dir)
    os.makedirs(consecutive_dir, exist_ok=True)

    png_pattern = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)
    existing_frames = []
    for fname in os.listdir(movie_frames_dir):
        match = png_pattern.match(fname)
        if match:
            existing_frames.append(int(match.group(1)))
    existing_frames.sort()

    if not existing_frames:
        print("No PNG overlays available for final movie. Skipping combine.")
        return

    # Copy overlays with updated 5-digit padding
    for i, original_fnum in enumerate(existing_frames, start=1):
        src = os.path.join(movie_frames_dir, f"frame_{original_fnum:05d}.png")
        dst = os.path.join(consecutive_dir, f"frame_{i:05d}.png")
        shutil.copy2(src, dst)

    # Update the ffmpeg input pattern to match 5-digit numbering
    cmd_combine = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-framerate", "30", "-start_number", "1", "-i", os.path.join(consecutive_dir, "frame_%05d.png"), "-pix_fmt", "yuv420p", "-vcodec", "libx264", mp4_output]
    subprocess.run(cmd_combine, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    done_msg = f"Movie saved to: {mp4_output}\n"
    last_messages.append(done_msg)
    print(done_msg)

def get_participant_names_from_datasheet_dir(datasheet_dir):
    participant_names = set()
    for filename in os.listdir(datasheet_dir):
        match = re.match(r"(.+?)-datasheet\.csv", filename, re.IGNORECASE)
        if match:
            participant_names.add(match.group(1))
    return sorted(list(participant_names))

def main():
    overall_start_time = time.time()
    datasheet_dir = MOVIE_DATASHEET_CSV_DIR  # get datasheet directory from config
    participant_names = get_participant_names_from_datasheet_dir(datasheet_dir)  # get participant names from datasheet directory

    if not participant_names:
        raise FileNotFoundError(f"No participant datasheet CSV files found in: {datasheet_dir}")

    for participant_name in participant_names:
        process_participant_dir(participant_name, overall_start_time)

if __name__ == "__main__":
    main()
