#!/usr/bin/env python3
# 5_datasheet_background.py

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import warnings

# Suppress specific pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

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
    """
    print("\n=== Segment and Trial Statistics ===")
    for (event, trial), group in df.groupby(['events_corrected', 'trial_number']):
        if event == 'green_dot':
            print(f"\nEvent: {event}, Trial: {trial}")
            print(f"  Total Frames: {group['frame_count'].max()}")
            print(f"  CHAPTER: n/a, frame_count_chapter: {group['frame_count_chapter'].tolist()}")
        else:
            print(f"\nEvent: {event}, Trial: {trial}")
            total_frames = group['frame_count'].max()
            print(f"  Total Frames: {total_frames}")
            chapters = group['CHAPTER'].value_counts().sort_index()
            for chap in ['approach', 'interaction', 'departure']:
                count = chapters.get(chap, 0)
                print(f"    {chap.capitalize()}: {count} frames")
    print("=== End of Statistics ===\n")

def process_events(df):
    """
    Process the DataFrame to ensure all events between green_dots are the same.
    """
    # Step 1: Identify if the event is a green_dot
    df['is_green_dot'] = df['events'] == 'green_dot'

    # Step 2: Identify where the event type changes
    df['segment_change'] = (df['is_green_dot'] != df['is_green_dot'].shift(1).fillna(False)).astype(bool)

    # Step 3: Assign segment number
    df['segment'] = df['segment_change'].cumsum()

    # Step 4: Drop helper columns
    df = df.drop(columns=['is_green_dot', 'segment_change'])

    # Step 5: Calculate mode for non-green_dot events
    df_non_green = df[df['events'] != 'green_dot'].copy()
    segment_modes = df_non_green.groupby('segment')['events'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'None'
    ).reset_index()
    segment_modes = segment_modes.rename(columns={'events': 'segment_mode'})

    # Step 6: Merge modes back
    df = df.merge(segment_modes, on='segment', how='left')

    # Step 7: Assign corrected events
    df['events_corrected'] = df.apply(
        lambda row: row['segment_mode'] if row['events'] != 'green_dot' else 'green_dot',
        axis=1
    )

    # Step 8: Assign frame count
    df['frame_count'] = df.groupby('segment').cumcount() + 1

    # Define event trial frame counts
    EVENT_TRIAL_FRAME_COUNT = {
        'sw': 150, 'swo': 185, 'hw': 150, 'hwo': 150,
        'gw': 150, 'gwo': 150, 'uhw': 150, 'uhwo': 150,
        'f': 150, 'ugw': 150, 'ugwo': 150
    }

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

    def assign_trials(group):
        if group['events_corrected'].iloc[0] == 'green_dot':
            group['trial_number'] = None
            group['frame_number_within_trial'] = group['frame_count']
            return group

        event_type = group['events_corrected'].iloc[0]
        expected_frames = EVENT_TRIAL_FRAME_COUNT.get(event_type, 150)
        total_frames = group['frame_count'].max()
        
        number_of_full_trials = total_frames // expected_frames
        leftover_frames = total_frames % expected_frames

        if leftover_frames >= (expected_frames / 2):
            number_of_trials = number_of_full_trials + 1
            trials_frame_counts = [expected_frames] * number_of_full_trials + [leftover_frames]
        else:
            if number_of_full_trials == 0:
                number_of_trials = 1
                trials_frame_counts = [leftover_frames]
            else:
                number_of_trials = number_of_full_trials
                trials_frame_counts = [expected_frames] * number_of_full_trials
                trials_frame_counts[-1] += leftover_frames

        trial_number = []
        frame_number_within_trial = []
        current_trial = 1
        for trial_frames in trials_frame_counts:
            for frame in range(1, trial_frames + 1):
                trial_number.append(current_trial)
                frame_number_within_trial.append(frame)
            current_trial += 1

        group = group.copy()
        group['trial_number'] = trial_number
        group['frame_number_within_trial'] = frame_number_within_trial
        return group

    def assign_chapters(trial_group):
        if trial_group['events_corrected'].iloc[0] == 'green_dot':
            trial_group['CHAPTER'] = 'n/a'
            trial_group['frame_count_chapter'] = trial_group['frame_number_within_trial']
            return trial_group

        event_type = trial_group['events_corrected'].iloc[0]
        chapter_mapping = EVENT_CHAPTER_FRAME_COUNT.get(
            event_type, 
            {'approach': 150, 'interaction': 150, 'departure': 150}
        )
        
        chapters = ['approach', 'interaction', 'departure']
        expected_frames = [chapter_mapping.get(chap, 150) for chap in chapters]

        cumulative_expected = []
        cum_sum = 0
        for ef in expected_frames:
            cum_sum += ef
            cumulative_expected.append(cum_sum)

        total_frames = trial_group['frame_number_within_trial'].max()

        chapters_assigned = []
        frame_counts_chapter = []
        for fn in trial_group['frame_number_within_trial']:
            assigned = False
            for i, threshold in enumerate(cumulative_expected):
                if fn <= threshold:
                    chap = chapters[i]
                    frame_num = fn - (cumulative_expected[i-1] if i > 0 else 0)
                    chapters_assigned.append(chap)
                    frame_counts_chapter.append(frame_num)
                    assigned = True
                    break
            if not assigned:
                chap = chapters[-1]
                frame_num = fn - (cumulative_expected[-1] if cumulative_expected else 0)
                chapters_assigned.append(chap)
                frame_counts_chapter.append(frame_num)

        trial_group = trial_group.copy()
        trial_group['CHAPTER'] = chapters_assigned
        trial_group['frame_count_chapter'] = frame_counts_chapter

        return trial_group

    # Apply trial assignments
    df = df.groupby('segment').apply(assign_trials)

    # Apply chapter assignments
    df = df.groupby(['events_corrected', 'trial_number']).apply(assign_chapters)
    
    # Final sorting
    df = df.sort_values('frame_number').reset_index(drop=True)

    return df

def save_corrected_csv(df, participant_dir, output_dir):
    """
    Save the corrected DataFrame to a new CSV file in the output directory.
    The filename is constructed as: (participant)-datasheet-background.csv
    For example, if participant_dir.name is "FiftySix-0501-1673-1024-inference",
    the output filename will be "FiftySix-0501-1673-datasheet-background.csv".
    """
    # First rename columns to the new names
    df = df.rename(columns={
        'frame_count': 'frame_count_event',
        'frame_number_within_trial': 'frame_count_trial_number',
        'CHAPTER': 'segment',
        'frame_count_chapter': 'frame_count_segment'
    })

    # Define the desired columns in the new order
    desired_columns = [
        'frame_number',
        'events_corrected',
        'frame_count_event',
        'trial_number',
        'frame_count_trial_number',
        'segment',
        'frame_count_segment'
    ]
    
    df_cleaned = df[desired_columns]
    
    # Construct the output filename
    participant_name = participant_dir.name
    # Remove '-1024' from the participant name if it exists
    participant_name = participant_name.replace("-1024", "")
    # Replace "inference" with "datasheet-background"
    output_filename = participant_name.replace("inference", "datasheet-background") + ".csv"
    corrected_csv_path = output_dir / output_filename
    try:
        df_cleaned.to_csv(corrected_csv_path, index=False)
        print(f"✓ Corrected CSV saved at: {corrected_csv_path}")
    except Exception as e:
        print(f"✗ Error saving corrected CSV: {e}")
        return

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

    # Rename columns (if necessary) – assuming CSV contains 'frame_number' and 'What'
    # Here we assume that the CSV has columns 'frame_number' and 'What' (which we treat as 'events')
    df = df.rename(columns={
        'frame_number': 'frame_number',
        'What': 'events'
    })

    df = df.sort_values('frame_number').reset_index(drop=True)

    # Process events
    df_corrected = process_events(df)
    
    # Print statistics
    print_statistics(df_corrected)

    # Drop unnecessary columns
    df_corrected = df_corrected.drop(columns=['segment', 'segment_mode', 'events'])
    
    # Save the corrected CSV to the global output directory with the new naming scheme
    save_corrected_csv(df_corrected, participant_dir, output_dir)

def main():
    """
    1. Look for detections_summary.csv files in participant output directories under
       D:\infant eye-tracking\paper-area\4_inference_background_output.
    2. For each participant, process the CSV to produce a corrected datasheet.
    3. Save the corrected datasheet in D:\infant eye-tracking\paper-area\5_datasheet_background_output
       with the filename (participant)-datasheet-background.csv.
    """
    # Input root is the global background inference output directory
    input_root = Path(r"D:\infant eye-tracking\paper-area\4_inference_background_output")
    # Global output directory for background datasheets
    output_root = Path(r"D:\infant eye-tracking\paper-area\5_datasheet_background_output")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Gather all participant directories by locating detections_summary.csv files
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
    
    # Define stacking order (priority) for events
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
