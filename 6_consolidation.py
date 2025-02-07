#!/usr/bin/env python3
import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def consolidate_data(datasheet_csv: Path, detections_csv: Path, output_path: Path):
    """
    Merges the YOLO datasheet CSV with the detections_summary_corrected CSV on frame number.
    The datasheet CSV (assumed to have a "Frame Number" column) is merged with the detections CSV
    (which should have a "frame_number" column). The resulting DataFrame is then written to the output path.
    """
    # Read the datasheet CSV (YOLO output)
    try:
        df_datasheet = pd.read_csv(datasheet_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df_datasheet = pd.read_csv(datasheet_csv, encoding="latin-1", errors="replace")
    
    # Read the detections CSV (background output)
    try:
        df_detections = pd.read_csv(detections_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df_detections = pd.read_csv(detections_csv, encoding="latin-1", errors="replace")
    
    # Merge the two DataFrames on frame number. The left merge key is "Frame Number" and the right is "frame_number".
    df_merged = pd.merge(
        df_datasheet,
        df_detections,
        left_on="Frame Number",
        right_on="frame_number",
        how="inner"
    )
    
    # Drop the duplicate frame number column if present.
    if "frame_number" in df_merged.columns:
        df_merged.drop(columns=["frame_number"], inplace=True)
    
    # Write the merged DataFrame to the output CSV.
    try:
        df_merged.to_csv(output_path, index=False)
        print(f"✓ Consolidated CSV saved: {output_path}")
    except PermissionError:
        print(f"Permission denied: {output_path}\nSkipping this participant.")

def main():
    # Define the fixed input and output directories.
    yolo_dir = Path(r"D:\infant eye-tracking\paper-area\3_datasheet_subjects_output")
    detections_dir = Path(r"D:\infant eye-tracking\paper-area\5_datasheet_background_output")
    output_dir = Path(r"D:\infant eye-tracking\paper-area\6_consolidation_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List all CSV files in the YOLO datasheet output directory.
    yolo_csv_files = list(yolo_dir.glob("*.csv"))
    if not yolo_csv_files:
        print(f"No YOLO CSV files found in {yolo_dir}")
        return
    
    processed_participants = []
    
    for yolo_csv in tqdm(yolo_csv_files, desc="Processing participants"):
        # The participant's identifier is assumed to be the filename stem (e.g., "FiftySix-0501-1673")
        participant_id = yolo_csv.stem
        
        # Construct the corresponding detections CSV filename
        # Changed to match the actual filename pattern
        detections_csv_filename = f"{participant_id}-background.csv"
        detections_csv = detections_dir / detections_csv_filename
        
        if not detections_csv.exists():
            print(f" - Detections CSV not found for participant {participant_id}: {detections_csv}")
            continue
        
        # The consolidated output file is named with the participant identifier.
        output_csv = output_dir / (participant_id + ".csv")
        
        # Skip processing if the output file already exists.
        if output_csv.exists():
            print(f" - Consolidated CSV already exists for {participant_id}, skipping.")
            processed_participants.append(participant_id)
            continue
        
        print(f"\nConsolidating data for participant: {participant_id}")
        print(f" - YOLO datasheet CSV: {yolo_csv}")
        print(f" - Detections CSV: {detections_csv}")
        consolidate_data(yolo_csv, detections_csv, output_csv)
        processed_participants.append(participant_id)
    
    if processed_participants:
        print("\nProcessed participants:")
        for p in processed_participants:
            print(f" - {p}")
    else:
        print("\nNo participants were processed.")

if __name__ == "__main__":
    main()