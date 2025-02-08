#!/usr/bin/env python3
# 5_csv_event.py

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import warnings

# Suppress specific pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Import configuration variables for step 5
from config import EVENT_CSV_INPUT_DIR, EVENT_CSV_OUTPUT_DIR

def locate_single_csv(run_dir, csv_name="detections_summary.csv"):
    """
    Locate a single CSV file with the specified name within the run directory.
    If multiple or no CSV files are found, return None.
    """
    csv_files = list(run_dir.glob(csv_name))
    if not csv_files:
        print(f"✗ No CSV file named '{csv_name}' found in {run_dir}. Skipping this directory.")
        return None
    elif len(csv_files) > 1:
        print(f"✗ Multiple CSV files named '{csv_name}' found in {run_dir}.")
        print("CSV Files Found:")
        for csv in csv_files:
            print(f" - {csv.name}")
        return None
    else:
        print(f"✓ CSV file located: {csv_files[0]}")
        return csv_files[0]

def extract_frame_number(filename):
    """
    Extracts the frame number from a filename.
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None

def calculate_time(frame_number, fps=30):
    """
    Converts frame number to time in seconds.
    """
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
    time_str = f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
    return total_seconds, time_str

def print_statistics(df):
    """
    Print detailed statistics about segments and trials.

    Expects columns:
    - 'event_verified'
    - 'frame_count_event'
    - 'trial_number_global'
    - 'segment' (the approach/interact/depart or 'n/a')
    """
    print("\n=== Segment and Trial Statistics ===")
    # Group by event_verified and trial_number_global
    for (evt, trial_g), group in df.groupby(['event_verified', 'trial_number_global']):
        print(f"\nEvent: {evt}, Global Trial: {trial_g}")
        total_frames = group['frame_count_event'].max()
        print(f"  Total Frames: {total_frames}")

        # Summarize the approach/interaction/departure chapters
        chapters = group['segment'].value_counts().sort_index()
        for chap_label in chapters.index:
            count = chapters[chap_label]
            print(f"    {chap_label.capitalize()}: {count} frames")
    print("=== End of Statistics ===\n")


def process_events(df):
    """
    Process the DataFrame to unify consecutive frames into segments,
    rename the event labels, assign trials, accumulate a global trial
    number for each event, and split frames into approach/interaction/
    departure chapters.

    Returns the DataFrame with:
      - 'event_verified' (renamed from events_corrected)
      - 'frame_count_event' (was frame_count)
      - 'frame_count_trial_number' (was frame_number_within_trial)
      - 'segment' (was CHAPTER => approach/interaction/departure)
      - 'frame_count_segment' (was frame_count_chapter)
      - 'trial_number'
      - 'trial_number_global'
    """
    # STEP 1: Mark green_dot vs. non-green_dot
    df['is_green_dot'] = df['events'] == 'green_dot'

    # STEP 2: Identify boundaries of segments
    df['segment_change'] = (df['is_green_dot'] != df['is_green_dot'].shift(1).fillna(False)).astype(bool)

    # STEP 3: Create a numeric segment ID
    df['segment_id'] = df['segment_change'].cumsum()

    # Drop helper columns
    df.drop(columns=['is_green_dot', 'segment_change'], inplace=True)

    # Find the most frequent event label in each segment (for non-green_dot).
    df_non_green = df[df['events'] != 'green_dot'].copy()
    segment_modes = (
        df_non_green.groupby('segment_id')['events']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'None')
        .reset_index()
        .rename(columns={'events': 'segment_mode'})
    )

    # Merge the "segment_mode" back in
    df = df.merge(segment_modes, on='segment_id', how='left')

    # If it's not green_dot, unify events to the mode
    df['events_corrected'] = df.apply(
        lambda row: row['segment_mode'] if row['events'] != 'green_dot' else 'green_dot',
        axis=1
    )

    # Count frames within each segment
    df['frame_count'] = df.groupby('segment_id').cumcount() + 1

    # Frame counts for each event
    EVENT_TRIAL_FRAME_COUNT = {
        'sw': 150, 'swo': 185, 'hw': 150, 'hwo': 150,
        'gw': 150, 'gwo': 185, 'uhw': 150, 'uhwo': 150, # GWO was 150, corrected to 185 to match old script
        'f': 150, 'ugw': 150, 'ugwo': 150
    }

    # Chapter splits
    EVENT_CHAPTER_FRAME_COUNT = {
        'gwo': {'approach': 40, 'interaction': 64, 'departure': 46},
        'uhw': {'approach': 32, 'interaction': 81, 'departure': 39},
        'uhwo': {'approach': 32, 'interaction': 85, 'departure': 33},
        'sw': {'approach': 39, 'interaction': 75, 'departure': 37},
        'swo': {'approach': 48, 'interaction': 87, 'departure': 50},
        'hw': {'approach': 32, 'interaction': 81, 'departure': 39},
        'hwo': {'approach': 32, 'interaction': 85, 'departure': 33},
        'ugw': {'approach': 31, 'interaction': 64, 'departure': 55},
        'ugwo': {'approach': 40, 'interaction': 64, 'departure': 46},
        'f': {'approach': 30, 'interaction': 82, 'departure': 38},
        'gw': {'approach': 31, 'interaction': 64, 'departure': 55}
    }

    def assign_local_trials(group):
        """Split each segment into local trials (1..N)."""
        if group['events_corrected'].iloc[0] == 'green_dot':
            group['trial_number'] = None
            group['frame_number_within_trial'] = group['frame_count']
            return group

        evt_type = group['events_corrected'].iloc[0]
        expected_frames = EVENT_TRIAL_FRAME_COUNT.get(evt_type, 150)
        total_frames = group['frame_count'].max()

        full_trials = total_frames // expected_frames
        leftover = total_frames % expected_frames

        # Decide how many trials
        if leftover >= (expected_frames / 2):
            number_of_trials = full_trials + 1
            trial_frames = [expected_frames] * full_trials + [leftover]
        else:
            if full_trials == 0:
                number_of_trials = 1
                trial_frames = [leftover]
            else:
                number_of_trials = full_trials
                trial_frames = [expected_frames] * full_trials
                trial_frames[-1] += leftover

        trial_number = []
        frame_within_trial = []
        current_trial = 1
        for tf in trial_frames:
            for fct in range(1, tf + 1):
                trial_number.append(current_trial)
                frame_within_trial.append(fct)
            current_trial += 1

        group = group.copy()
        group['trial_number'] = trial_number
        group['frame_number_within_trial'] = frame_within_trial
        return group

    def assign_chapters(trial_group):
        """Assign approach/interaction/departure based on frame_number_within_trial."""
        if trial_group['events_corrected'].iloc[0] == 'green_dot':
            trial_group['CHAPTER'] = 'n/a'
            trial_group['frame_count_chapter'] = trial_group['frame_number_within_trial']
            return trial_group

        evt_type = trial_group['events_corrected'].iloc[0]
        mapping = EVENT_CHAPTER_FRAME_COUNT.get(
            evt_type, {'approach': 150, 'interaction': 150, 'departure': 150}
        )

        chapters = ['approach', 'interaction', 'departure']
        expected_frames = [mapping.get(ch, 150) for ch in chapters]
        cumulative_thresholds = []
        csum = 0
        for ef in expected_frames:
            csum += ef
            cumulative_thresholds.append(csum)

        assigned_chaps = []
        frame_chap = []

        for fn in trial_group['frame_number_within_trial']:
            placed = False
            for i, th in enumerate(cumulative_thresholds):
                if fn <= th:
                    assigned_chaps.append(chapters[i])
                    offset = cumulative_thresholds[i - 1] if i > 0 else 0
                    frame_chap.append(fn - offset)
                    placed = True
                    break
            if not placed:
                assigned_chaps.append(chapters[-1])
                offset = cumulative_thresholds[-1]
                frame_chap.append(fn - offset)

        trial_group = trial_group.copy()
        trial_group['CHAPTER'] = assigned_chaps
        trial_group['frame_count_chapter'] = frame_chap
        return trial_group

    # Group by numeric segment ID
    df = df.groupby('segment_id').apply(assign_local_trials)

    # Then group by (events_corrected, local trial_number)
    df = df.groupby(['events_corrected', 'trial_number']).apply(assign_chapters)
    df.sort_values('frame_number', inplace=True, ignore_index=True)

    # Accumulate global trial numbers for each event
    df['trial_number_global'] = 0
    event_trial_offset = {}
    for seg_id, seg_data in df.groupby('segment_id', sort=False):
        e_type = seg_data['events_corrected'].iloc[0]
        if e_type == 'green_dot':
            continue

        offset = event_trial_offset.get(e_type, 0)
        df.loc[seg_data.index, 'trial_number_global'] = seg_data['trial_number'] + offset
        event_trial_offset[e_type] = offset + seg_data['trial_number'].max()

    # Final rename:
    #   events_corrected -> event_verified
    #   frame_count -> frame_count_event
    #   frame_number_within_trial -> frame_count_trial_number
    #   CHAPTER -> segment
    #   frame_count_chapter -> frame_count_segment
    df.rename(columns={
        'events_corrected': 'event_verified',
        'frame_count': 'frame_count_event',
        'frame_number_within_trial': 'frame_count_trial_number',
        'CHAPTER': 'segment',
        'frame_count_chapter': 'frame_count_segment'
    }, inplace=True)

    return df

def save_corrected_csv(df, participant_dir, output_dir):
    """
    Save the corrected DataFrame to a new CSV file in the output directory.
    The filename is constructed as: (participant)-datasheet-background.csv
    """
    desired_columns = [
        'frame_number',
        'event_verified',
        'frame_count_event',
        'trial_number',
        'trial_number_global',
        'frame_count_trial_number',
        'segment',
        'frame_count_segment'
    ]
    df_cleaned = df[[c for c in desired_columns if c in df.columns]]

    participant_name = participant_dir.name
    participant_name = participant_name.replace("-1024", "")
    output_filename = participant_name.replace("inference", "datasheet-background") + ".csv"
    corrected_csv_path = output_dir / output_filename
    try:
        df_cleaned.to_csv(corrected_csv_path, index=False)
        print(f"✓ Corrected CSV saved at: {corrected_csv_path}")
    except Exception as exc:
        print(f"✗ Error saving corrected CSV: {exc}")


def process_participant(participant_dir, priority_order, fps, output_dir):
    """
    Process a single participant directory.
    participant_dir: Directory containing detections_summary.csv.
    output_dir: Global output directory for background datasheets.
    """
    csv_file = locate_single_csv(participant_dir)
    if not csv_file:
        return

    try:
        df = pd.read_csv(csv_file)
        print(f"✓ CSV loaded successfully from {csv_file}. Total frames: {len(df)}")
    except Exception as e:
        print(f"✗ Error reading CSV file {csv_file}: {e}. Skipping.")
        return

    df = df.rename(columns={
        'frame_number': 'frame_number',
        'What': 'events'
    })

    df = df.sort_values('frame_number').reset_index(drop=True)
    df_corrected = process_events(df)

    print_statistics(df_corrected)
    df_corrected = df_corrected.drop(columns=['segment_id', 'segment_mode', 'events']) # Removed 'segment' and 'segment_mode'
    save_corrected_csv(df_corrected, participant_dir, output_dir)


def main():
    """
    1. Look for detections_summary.csv files in participant output directories under
       the background inference output directory.
    2. For each participant, process the CSV to produce a corrected datasheet.
    3. Save the corrected datasheet in the global output directory for background datasheets.
    """
    input_root = Path(EVENT_CSV_INPUT_DIR) # Use corrected input path from config
    output_root = Path(EVENT_CSV_OUTPUT_DIR) # Use corrected output path from config
    output_root.mkdir(parents=True, exist_ok=True)

    participant_dirs = set()
    for path in input_root.rglob("detections_summary.csv"):
        participant_dirs.add(path.parent)

    if not participant_dirs:
        print("No directories with detections_summary.csv found in the background inference outputs.")
        print("Please ensure you have run the background inference script first.")
        return

    print(f"Found {len(participant_dirs)} directories to process:")
    for d in participant_dirs:
        print(f"- {d}")

    priority_order = [
        'toy',
        'hand_man',
        'hand_woman',
        'face_woman',
        'face_man',
        'body_woman',
        'body_man',
        'green_dot',
        'screen'
    ]

    for participant_dir in tqdm(list(participant_dirs), desc="Processing participants"):
        print(f"\n--- Processing Directory: {participant_dir} ---")
        process_participant(participant_dir, priority_order, fps=30, output_dir=output_root)

    print("\n✓ All directories processed successfully!")


if __name__ == "__main__":
    main()