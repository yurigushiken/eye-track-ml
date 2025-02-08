# config.py

from pathlib import Path

# --------------------------------------------------
# Configurations for 1_preprocessing.py (Step 1: Video Preprocessing)
#   Input:      VIDEO_DIR (Raw video files)
#   Output:     PREPROCESSING_OUTPUT_DIR (Folder for preprocessed videos/frames)
# --------------------------------------------------
VIDEO_DIR = Path(r"D:\infant eye-tracking\paper-area\0_participant_videos")
PREPROCESSING_OUTPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\1_preprocessing_output")

# --------------------------------------------------
# Configurations for 2_inference_object_yolo.py (Step 2: YOLO Object Inference)
#   Input:      YOLO_OBJ_INPUT_DIR (Directory with preprocessed frames for YOLO object detection)
#   Output:     YOLO_OBJ_OUTPUT_DIR (Folder to store YOLO object detection results)
# --------------------------------------------------
YOLO_OBJ_INPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\1_preprocessing_output")
YOLO_OBJ_OUTPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\2_inference_objects_yolo_output")

# --------------------------------------------------
# Configurations for 2_inference_objects_yolo_sam.py (Step 2: YOLO & SAM2 Inference)
#   Input:      YOLO_SAM_INPUT_DIR (Directory with preprocessed frames for YOLO & SAM2 inference)
#   Output:     YOLO_SAM_OUTPUT_DIR (Folder to store YOLO & SAM2 inference results)
#   SAM2 Specific:
#       SAM2_CONFIG_FILE: Path to SAM2 configuration file
#       SAM2_CHECKPOINT:  Path to SAM2 checkpoint file
# --------------------------------------------------
YOLO_SAM_INPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\1_preprocessing_output")
YOLO_SAM_OUTPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\2_inference_objects_yolo_sam_output")
SAM2_CONFIG_FILE = Path(r"C:/Users/yurig/Desktop/Infant Eye Tracking (desktop)/our own software/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml")
SAM2_CHECKPOINT = Path(r"C:/Users/yurig/Desktop/Infant Eye Tracking (desktop)/our own software/sam2/checkpoints/checkpoint-b4.pt")

# --------------------------------------------------
# Configurations for 3_csv_objects_yolo_sam.py (Step 3: Create CSV from YOLO & SAM Detections)
#   Input:      CSV_YOLO_SAM_INPUT_DIR (Directory with YOLO & SAM object detection results)
#   Output:     CSV_YOLO_SAM_OUTPUT_DIR (Folder to store the CSV datasheet for YOLO & SAM detections)
# --------------------------------------------------
CSV_YOLO_SAM_INPUT_DIR = YOLO_SAM_OUTPUT_DIR
CSV_YOLO_SAM_OUTPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\3_csv_objects_yolo_sam_output")

# --------------------------------------------------
# Configurations for 3_csv_objects.py (Step 3: Create CSV from YOLO Detections) # Renamed to avoid conflict, but not used now.
#   Input:      CSV_YOLO_INPUT_DIR (Directory with YOLO object detection results)
#   Output:     CSV_YOLO_OUTPUT_DIR (Folder to store the CSV datasheet for YOLO detections)
# --------------------------------------------------
CSV_YOLO_INPUT_DIR = YOLO_OBJ_OUTPUT_DIR # Still defined, but likely not used in current workflow if using YOLO_SAM
CSV_YOLO_OUTPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\3_csv_objects_yolo_output") # Still defined, but likely not used in current workflow if using YOLO_SAM

# --------------------------------------------------
# Configurations for 4_inference_event.py (Step 4: Event Classification Inference)
#   Input:      EVENT_INFERENCE_INPUT_DIR (Directory with frames for event inference)
#   Output:     EVENT_INFERENCE_OUTPUT_DIR (Folder to store event classification results)
#   Additional: CLASSIFICATION_INFERENCE_URL (URL for event inference server)
# --------------------------------------------------
EVENT_INFERENCE_INPUT_DIR = PREPROCESSING_OUTPUT_DIR
EVENT_INFERENCE_OUTPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\4_inference_background_output")
CLASSIFICATION_INFERENCE_URL = "http://localhost:9001/infer/classification"

# --------------------------------------------------
# Configurations for 5_csv_event.py (Step 5: Create CSV for Event Inference)
#   Input:      EVENT_CSV_INPUT_DIR (Directory with event inference results)
#   Output:     EVENT_CSV_OUTPUT_DIR (Folder to store the CSV datasheet for event inference)
# --------------------------------------------------
EVENT_CSV_INPUT_DIR = EVENT_INFERENCE_OUTPUT_DIR
EVENT_CSV_OUTPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\5_datasheet_background_output")

# --------------------------------------------------
# Configurations for 6_csv_consolidation.py (Step 6: Consolidate Data)
#   Input:      CSV_YOLO_OUTPUT_DIR (CSV datasheet from YOLO detections) # Note: May need to be updated if using YOLO_SAM output instead
#               EVENT_CSV_OUTPUT_DIR (CSV datasheet from event inference)
#   Output:     CONSOLIDATION_OUTPUT_DIR (Folder to store the consolidated CSV data)
# --------------------------------------------------
CONSOLIDATION_OUTPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\6_consolidation_output")

# --------------------------------------------------
# Configurations for 7_movie.py (Step 7: Create Movie)
#   Input:      MOVIE_FRAMES_BASE_DIR (Base directory for image frames)
#               MOVIE_DATASHEET_CSV_DIR (Directory for datasheet CSV files)
#               MOVIE_PARTICIPANT_CSV_DIR (Directory for participant CSV files)
#   Output:     MOVIE_OUTPUT_DIR (Base directory for output movies - movies will be in participant subfolders here)
#   Additional: MOVIE_FONT_PATH (Font used for text overlays in the movie)
# --------------------------------------------------
MOVIE_FRAMES_BASE_DIR = Path(r"D:\infant eye-tracking\paper-area\2_inference_objects_yolo_output") # Directory containing image frames
MOVIE_DATASHEET_CSV_DIR = Path(r"D:\infant eye-tracking\paper-area\3_csv_objects_yolo_output") # Directory containing datasheet CSV files
MOVIE_PARTICIPANT_CSV_DIR = Path(r"D:\infant eye-tracking\paper-area\6_consolidation_output") # Directory containing participant CSV files
MOVIE_OUTPUT_DIR = Path(r"D:\infant eye-tracking\paper-area\7_movie_output") # Base output directory for movies - movies will be in participant subfolders here
MOVIE_FONT_PATH = Path(r"C:/Windows/Fonts/times.ttf")