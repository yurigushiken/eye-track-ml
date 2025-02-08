#!/usr/bin/env python3
# 6_csv_consolidation.py

import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import configuration variables for consolidation
from config import CSV_YOLO_OUTPUT_DIR, EVENT_CSV_OUTPUT_DIR, CONSOLIDATION_OUTPUT_DIR

# Dictionaries for parsing spelled-out numbers. (From OLD SCRIPT - No changes needed)
UNITS = {
    "Zero": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
    "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10,
    "Eleven": 11, "Twelve": 12, "Thirteen": 13, "Fourteen": 14,
    "Fifteen": 15, "Sixteen": 16, "Seventeen": 17, "Eighteen": 18,
    "Nineteen": 19
}
TENS = {
    "Twenty": 20, "Thirty": 30, "Forty": 40, "Fourty": 40,
    "Fifty": 50, "Sixty": 60, "Seventy": 70, "Eighty": 80,
    "Ninety": 90
}

def parse_spelled_number(spelled_str: str) -> int:
    """
    Splits a spelled-out number (like 'FiftySix') into parts
    and sums them. E.g., 'FiftySix' -> 56, 'FourtyOne' -> 41.
    (From OLD SCRIPT - No changes needed)
    """
    parts = re.findall(r'[A-Z][a-z]+', spelled_str)  # Break on capital letters
    total = 0
    for p in parts:
        if p in TENS:
            total += TENS[p]
        elif p in UNITS:
            total += UNITS[p]
    return total

def get_participant_type_and_age_months(yolo_csv_path: Path) -> (str, int):
    """
    Determines participant type and age from the YOLO CSV filename.
    Parses participant ID from the filename.
    If the parsed number is over 18, it's an adult (age in years).
    If the parsed number is 18 or less, it's an infant (age in months).
    (MODIFIED LOGIC as per user request)
    """
    filename = yolo_csv_path.name
    participant_id = filename.replace("-datasheet.csv", "") # Extract participant ID from filename
    first_chunk = participant_id.split("-")[0] # Get first part for spelled number parsing
    spelled_number_value = parse_spelled_number(first_chunk)

    if spelled_number_value > 18:
        return "adult", spelled_number_value * 12 # Age in years, convert to months
    else:
        return "infant", spelled_number_value      # Age in months

def consolidate_data(datasheet_csv: Path, detections_csv: Path, output_path: Path, participant_type: str, age_months: int):
    """
    Merges datasheet CSV with detections_summary_corrected.csv on frame number.
    Columns from the datasheet appear on the left, columns from detections on the right.
    Adds columns for participant_type, participant_age_months, and participant_age_years.
    (Modified NEW SCRIPT to include participant info from OLD SCRIPT - No changes needed here)
    """
    try:
        df_datasheet = pd.read_csv(datasheet_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df_datasheet = pd.read_csv(datasheet_csv, encoding="latin-1", errors="replace")
    try:
        df_detections = pd.read_csv(detections_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df_detections = pd.read_csv(detections_csv, encoding="latin-1", errors="replace")
    df_merged = pd.merge(
        df_datasheet,
        df_detections,
        left_on="Frame Number",
        right_on="frame_number",
        how="inner"
    )
    if "frame_number" in df_merged.columns:
        df_merged.drop(columns=["frame_number"], inplace=True)

    # Add participant type and age columns (From OLD SCRIPT - No changes needed here)
    df_merged["participant_type"] = participant_type
    df_merged["participant_age_months"] = age_months
    df_merged["participant_age_years"] = (df_merged["participant_age_months"] / 12).round(1)


    try:
        df_merged.to_csv(output_path, index=False)
        print(f"✓ Consolidated CSV saved: {output_path}")
    except PermissionError:
        print(f"Permission denied: {output_path}\nSkipping this participant.")

def main():
    yolo_dir = CSV_YOLO_OUTPUT_DIR
    detections_dir = EVENT_CSV_OUTPUT_DIR
    output_dir = CONSOLIDATION_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_csv_files = list(yolo_dir.glob("*.csv"))
    if not yolo_csv_files:
        print(f"No YOLO CSV files found in {yolo_dir}")
        return
    processed_participants = []
    for yolo_csv in tqdm(yolo_csv_files, desc="Processing participants"):
        participant_id = yolo_csv.stem

        # Get participant type and age from the YOLO CSV FILENAME (MODIFIED LOGIC)
        p_type, p_age_months = get_participant_type_and_age_months(yolo_csv)

        detections_csv_filename = f"{participant_id}-background.csv"
        detections_csv = detections_dir / detections_csv_filename
        if not detections_csv.exists():
            print(f" - Detections CSV not found for participant {participant_id}: {detections_csv}")
            continue
        # Change output filename to use "-maindoc.csv"
        output_csv = output_dir / (participant_id.replace("-datasheet", "") + "-maindoc.csv")
        if output_csv.exists():
            print(f" - Consolidated CSV already exists for {participant_id}, skipping.")
            processed_participants.append(participant_id)
            continue
        print(f"\nConsolidating data for participant: {participant_id}")
        print(f" - YOLO datasheet CSV: {yolo_csv}")
        print(f" - Detections CSV: {detections_csv}")
        consolidate_data(yolo_csv, detections_csv, output_csv, p_type, p_age_months) # Pass participant info
        processed_participants.append(participant_id)
    if processed_participants:
        print("\nProcessed participants:")
        for p in processed_participants:
            print(f" - {p}")
    else:
        print("\nNo participants were processed.")

if __name__ == "__main__":
    main()