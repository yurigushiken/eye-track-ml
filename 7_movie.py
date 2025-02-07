#!/usr/bin/env python3
# Below is the full revised script with only the requested modifications:

# Check if movie.csv already exists. If it does, skip re-creating it.
# Check if all frames in movie_frames already exist and the final movie .mp4 file already exists. If both conditions are true, skip making a new movie.
# Otherwise, proceed with frame creation for any missing frames and, if needed, combine them into the final .mp4.

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

# Adjust if needed
FONT_PATH_SERIF = r"C:/Windows/Fonts/times.ttf"


def read_csv_with_fallback_encodings(filepath, fallback_encodings=None):
    """
    Attempts to read the CSV with a list of encodings.
    Returns a list of dict rows on success. Raises UnicodeDecodeError on complete failure.
    """
    if fallback_encodings is None:
        # You can adjust or reorder these as needed
        fallback_encodings = ["utf-8", "latin-1", "cp1252"]

    for enc in fallback_encodings:
        try:
            with open(filepath, mode="r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            # Try the next encoding
            pass

    raise UnicodeDecodeError(
        f"Could not decode file {filepath} with any of the tried encodings: {fallback_encodings}"
    )


def escape_drawtext(text):
    """Escapes special characters for ffmpeg drawtext."""
    if not text:
        return ""
    text = text.replace(':', '\\:')
    text = text.replace("'", "\\'")
    return text


def frame_to_timecode(frame_num, fps=30):
    """
    Returns total elapsed time in H:MM:SS.ss format,
    continuing from 0 up to total runtime without resetting each minute.
    """
    total_seconds = frame_num / float(fps)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def is_hidden_or_resource_file(fname):
    """
    Returns True if the filename begins with '.' or '._'
    which is often a hidden or resource fork file (especially from macOS).
    """
    return fname.startswith('.') or fname.startswith('._')


def find_datasheet_csv_in_participant_dir(participant_dir):
    """
    Returns the path of the *first* CSV file found directly in participant_dir 
    (ignoring subfolders), that does not begin with a dot (.) or '._'.
    This is considered the 'datasheet' CSV.

    If no valid CSV is found, returns None.
    """
    for fname in os.listdir(participant_dir):
        # Skip if not .csv or hidden
        if not fname.lower().endswith(".csv"):
            continue
        if is_hidden_or_resource_file(fname):
            continue

        full_path = os.path.join(participant_dir, fname)
        if os.path.isfile(full_path):
            return full_path

    return None


def find_participant_csv(participant_dir):
    """
    Finds a CSV in the participant_dir (ignoring subfolders),
    excluding any file name that has 'datasheet' in it (case-insensitive).
    Also skips hidden/resource files like '._*.csv'.

    Returns the first match or None if none found.
    """
    for fname in os.listdir(participant_dir):
        if not fname.lower().endswith(".csv"):
            continue
        if is_hidden_or_resource_file(fname):
            continue
        if "datasheet" in fname.lower():
            continue

        full_path = os.path.join(participant_dir, fname)
        if os.path.isfile(full_path):
            return full_path

    return None


def read_merged_data(participant_dir):
    """
    1) Finds the 'datasheet' CSV in participant_dir (the first CSV it sees at top-level).
    2) Finds the 'participant' CSV in participant_dir (the first CSV not containing 'datasheet').
    3) Reads the datasheet CSV data and participant CSV data.
       - The datasheet CSV must have 'Frame Number' column.
       - The participant CSV provides 'What' and 'Where' columns (plus participant metadata).
    4) Merges them so that 'What' and 'Where' from the participant CSV overwrite those from datasheet.
    """

    datasheet_csv = find_datasheet_csv_in_participant_dir(participant_dir)
    if not datasheet_csv:
        raise FileNotFoundError(
            f"No valid (non-hidden) CSV found in participant directory: {participant_dir} "
            f"(for datasheet). Skipping any files like '._*.csv'."
        )

    if not os.path.exists(datasheet_csv):
        raise FileNotFoundError(f"Datasheet CSV not found at path: {datasheet_csv}")

    # Read datasheet CSV
    datasheet_rows = read_csv_with_fallback_encodings(datasheet_csv)
    if not datasheet_rows:
        raise ValueError(f"CSV '{datasheet_csv}' has no data rows.")

    # Ensure 'Frame Number' exists
    if "Frame Number" not in datasheet_rows[0]:
        raise ValueError(f"CSV '{datasheet_csv}' must contain 'Frame Number' column.")

    # Convert datasheet to dict keyed by frame number
    datasheet_data = {}
    for row in datasheet_rows:
        frame_str = row.get("Frame Number", "").strip()
        if frame_str.isdigit():
            datasheet_data[int(frame_str)] = row

    # Now read the participant CSV (the one that doesn't have 'datasheet' in name)
    participant_csv = find_participant_csv(participant_dir)
    participant_data = {}
    participant_type = None
    participant_age_months = None
    participant_age_years = None

    if participant_csv and os.path.exists(participant_csv):
        p_rows = read_csv_with_fallback_encodings(participant_csv)
        # It's not necessarily an error if participant CSV has no data:
        if p_rows:
            for p_row in p_rows:
                frame_str_p = p_row.get("Frame Number", "").strip()
                if frame_str_p.isdigit():
                    participant_data[int(frame_str_p)] = p_row
                # Capture participant metadata
                if participant_type is None and "participant_type" in p_row:
                    participant_type = p_row["participant_type"].strip()
                if participant_age_months is None and "participant_age_months" in p_row:
                    participant_age_months = p_row["participant_age_months"].strip()
                if participant_age_years is None and "participant_age_years" in p_row:
                    participant_age_years = p_row["participant_age_years"].strip()

    # Merge: datasheet row first, then overwrite with participant row (esp. "What" and "Where")
    csv_data_by_frame = {}
    all_frames = set(datasheet_data.keys()).union(set(participant_data.keys()))
    for fnum in sorted(all_frames):
        merged_row = {}
        if fnum in datasheet_data:
            merged_row.update(datasheet_data[fnum])
        if fnum in participant_data:
            p_row = participant_data[fnum]
            # Overwrite "What" and "Where"
            merged_row["What"] = p_row.get("What", merged_row.get("What", ""))
            merged_row["Where"] = p_row.get("Where", merged_row.get("Where", ""))
            # Copy all other participant columns that might be relevant
            for key, val in p_row.items():
                if key not in ["Frame Number", "What", "Where"]:
                    merged_row[key] = val
        merged_row["Frame Number"] = fnum
        csv_data_by_frame[fnum] = merged_row

    return csv_data_by_frame, participant_type, participant_age_months, participant_age_years


def create_movie_csv(participant_dir, csv_data_by_frame):
    """
    Writes a 'movie.csv' with needed columns for reference/debug.
    """
    movie_csv_path = os.path.join(participant_dir, "movie.csv")
    fieldnames = [
        "Participant", "Frame Number", "Time", "What", "Where",
        "Onset", "Offset", "Blue Dot Center", "event_verified",
        "frame_count_event", "trial_number_global", "frame_count_trial_number",
        "segment", "frame_count_segment", "participant_type",
        "participant_age_months", "participant_age_years"
    ]

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
    """
    Finds all filenames matching 'frame_XXXX_annotated.jpg' -> returns list of int(XXXX).
    """
    frame_nums = []
    pattern = re.compile(r"^frame_(\d+)_annotated\.jpg$", re.IGNORECASE)
    if os.path.isdir(images_dir):
        for fname in os.listdir(images_dir):
            match = pattern.match(fname)
            if match:
                frame_nums.append(int(match.group(1)))
    return sorted(frame_nums)


def process_participant_dir(participant_dir, overall_start_time):
    """
    Reads and merges CSV data (datasheet CSV + participant CSV in participant_dir),
    overlays info onto annotated frames, and produces a final .mp4 in participant_dir,
    *unless* that final .mp4 (and all .png frames) already exist and movie.csv is present.
    """

    participant_folder_name = os.path.basename(participant_dir)

    # images_dir is typically: participant_dir/output_yolo/image_frames-<foldername>
    images_dir = os.path.join(
        participant_dir, 
        "output_yolo", 
        f"image_frames-{participant_folder_name}"
    )

    # Determine final MP4 name by matching any .avi in participant_dir
    avi_files = [f for f in os.listdir(participant_dir) if f.lower().endswith('.avi')]
    if avi_files:
        avi_file_base = os.path.splitext(avi_files[0])[0]
        final_movie_name = f"{avi_file_base}-movie.mp4"
    else:
        final_movie_name = "movie.mp4"

    mp4_output = os.path.join(participant_dir, final_movie_name)

    # Merge data
    csv_data_by_frame, participant_type, participant_age_months, participant_age_years = read_merged_data(participant_dir)

    # Check if movie.csv already exists, if so, skip re-creating it
    movie_csv_path = os.path.join(participant_dir, "movie.csv")
    if not os.path.exists(movie_csv_path):
        create_movie_csv(participant_dir, csv_data_by_frame)
    else:
        print(f"movie.csv already exists in {participant_dir}. Skipping creation.")

    # If the images_dir doesn't exist, there's nothing to process
    if not os.path.isdir(images_dir):
        print(f"No images folder found in: {images_dir}. Skipping.")
        return

    all_frame_nums = gather_frame_numbers_in_inference_dir(images_dir)
    if not all_frame_nums:
        print(f"No annotated frames found in {images_dir}. Skipping.")
        return

    # Prepare output folder for the final overlay frames in the participant directory
    movie_frames_dir = os.path.join(participant_dir, "movie_frames")
    os.makedirs(movie_frames_dir, exist_ok=True)

    # Gather which frames we already have in movie_frames_dir
    existing_overlay_pattern = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)
    existing_overlay_frames = set()
    for fname in os.listdir(movie_frames_dir):
        match = existing_overlay_pattern.match(fname)
        if match:
            existing_overlay_frames.add(int(match.group(1)))

    # Identify frames that need to be created
    missing_frames = [f for f in all_frame_nums if f not in existing_overlay_frames]

    total_frames = len(all_frame_nums)
    start_time_dir = time.time()
    last_messages = []

    def print_dynamic_status(current_frame_index, total_frames):
        """Updates a status line on the console for each processed frame."""
        print("\033[H\033[J", end="")  # Clears screen (for a dynamic effect in many terminals)
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

    # -------------
    # OVERLAY PHASE (only for missing frames)
    # -------------
    if missing_frames:
        print(f"\n=== Overlay Creation Phase for {participant_folder_name} ===")
        for i, fnum in enumerate(missing_frames, start=1):
            out_filename = f"frame_{fnum:04d}.png"
            out_path = os.path.join(movie_frames_dir, out_filename)

            # If the output file already exists (just in case), skip it
            if os.path.exists(out_path):
                progress_msg = f"Frame {fnum:04d} already exists. Skipping overlay."
                last_messages.append(progress_msg)
                print_dynamic_status(i, len(missing_frames))
                continue

            image_name = f"frame_{fnum:04d}_annotated.jpg"
            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                progress_msg = f"Missing {image_name}; skipping."
                last_messages.append(progress_msg)
                print_dynamic_status(i, len(missing_frames))
                continue

            # Get row data
            row = csv_data_by_frame.get(fnum, {})

            # Start with a basic crop if needed
            filter_str = "crop=900:700:(in_w-900)/2:(in_h-700)/2"

            # LEFT OVERLAY
            if SHOW_LEFT_OVERLAY:
                frame_label = f"{fnum} / {all_frame_nums[-1]}"
                what_value = row.get("What", "").strip().lower()
                where_value = row.get("Where", "").strip().lower()
                top_text = f"{what_value}, {where_value}"
                frame_label_esc = escape_drawtext(frame_label)
                top_text_esc = escape_drawtext(top_text)
                filter_str += (
                    f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                    f"text='{frame_label_esc}':x=20:y=20:"
                    f"fontcolor=white:fontsize=24,"
                    f"drawtext=fontfile='{FONT_PATH_SERIF}':text='{top_text_esc}':"
                    f"x=(w-text_w)/2:y=40:fontcolor=white:fontsize=72"
                )

            # RIGHT OVERLAY
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

                filter_str += (
                    f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                    f"text='{time_str_esc}':x=(w-text_w)-20:y=20:"
                    f"fontcolor=white:fontsize=24,"
                    f"drawtext=fontfile='{FONT_PATH_SERIF}':"
                    f"text='{event_verified_esc}':"
                    f"x=(w-text_w)-20:y=50:fontcolor=white:fontsize=24,"
                    f"drawtext=fontfile='{FONT_PATH_SERIF}':"
                    f"text='{segment_esc}':"
                    f"x=(w-text_w)-20:y=80:fontcolor=white:fontsize=24"
                )

            # BOTTOM OVERLAY
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
                filter_str += (
                    f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                    f"text='{participant_text_esc}':"
                    f"x=(w-text_w)/2:y=(h-text_h)-60:"
                    f"fontcolor=white:fontsize=28"
                )

            # Example of an extra bottom-right overlay
            bottom_right_text = escape_drawtext("verified in part with symbolic system")
            filter_str += (
                f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                f"text='{bottom_right_text}':"
                f"x=(w-text_w)-20:y=(h-text_h)-20:"
                f"fontcolor=white:fontsize=16"
            )

            progress_msg = f"Processing frame {fnum:04d} -> {out_filename}"
            last_messages.append(progress_msg)
            print_dynamic_status(i, len(missing_frames))

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "warning",
                "-y",
                "-i", image_path,
                "-vf", filter_str,
                out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        print(f"All overlay frames already exist for {participant_folder_name}. Skipping overlay creation.")

    # -------------
    # COMBINE PHASE
    # -------------
    # If the final MP4 already exists (and presumably is correct), we can skip combining
    if os.path.exists(mp4_output):
        # Double-check if we've got the correct total number of frames (simple check).
        # If there are no missing frames, assume the final movie is complete.
        if not missing_frames:
            print(f"Final movie {mp4_output} already exists and no frames are missing. Skipping combine.")
            return
        else:
            print(f"Final movie {mp4_output} exists but new frames were added. Recreating the final movie.")
            # We will continue to combine below
    else:
        print(f"\n=== Combine Phase for {participant_folder_name} ===")

    combination_msg = f"Collecting overlays from {movie_frames_dir} for final movie..."
    last_messages.append(combination_msg)

    # Prepare a fresh subfolder for consecutive frames
    consecutive_dir = os.path.join(movie_frames_dir, "consecutive")
    if os.path.isdir(consecutive_dir):
        shutil.rmtree(consecutive_dir)
    os.makedirs(consecutive_dir, exist_ok=True)

    # Gather all existing overlay PNGs
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

    # Copy them in a strictly consecutive pattern
    for i, original_fnum in enumerate(existing_frames, start=1):
        src = os.path.join(movie_frames_dir, f"frame_{original_fnum:04d}.png")
        dst = os.path.join(consecutive_dir, f"frame_{i:04d}.png")
        shutil.copy2(src, dst)

    # Now combine the consecutive frames into an MP4
    cmd_combine = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-framerate", "30",
        "-start_number", "1",
        "-i", os.path.join(consecutive_dir, "frame_%04d.png"),
        "-pix_fmt", "yuv420p",
        "-vcodec", "libx264",
        mp4_output
    ]
    subprocess.run(cmd_combine, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    done_msg = f"Movie saved to: {mp4_output}\n"
    last_messages.append(done_msg)
    print(done_msg)


def find_participant_main_dirs(base_path):
    """
    Recursively searches for directories that contain 'output_yolo'.
    Those directories are considered participant-level directories.
    """
    participant_dirs = set()
    for root, dirs, files in os.walk(base_path):
        dirs_lower = [d.lower() for d in dirs]
        if "output_yolo" in dirs_lower:
            participant_dirs.add(root)
    return sorted(list(participant_dirs))


def main():
    overall_start_time = time.time()
    # Adjust base_path as you need:
    base_path = r"D:\infant eye-tracking\DATA_PROCESSING\(2025-01-30)"

    # Get all participant directories (those containing "output_yolo")
    participant_dirs = find_participant_main_dirs(base_path)
    if not participant_dirs:
        raise FileNotFoundError("No participant directories found with 'output_yolo' subfolders.")

    # Process each participant directory
    for p_dir in participant_dirs:
        process_participant_dir(p_dir, overall_start_time)


if __name__ == "__main__":
    main()
