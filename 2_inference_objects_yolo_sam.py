#!/usr/bin/env python3
# 2_inference_objects_yolo_sam.py

import os
import cv2
import json
import time
import base64
import torch
import requests
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from torch.utils.data import DataLoader, Dataset
from itertools import islice
from collections import deque

import requests.packages.urllib3.util.retry
import requests.adapters

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dotenv import load_dotenv
load_dotenv()

##############################################################################
#                                  SETTINGS                                  #
##############################################################################

# For this script, the input and output are defined separately for the YOLO+SAM2 inference.
from config import YOLO_SAM_INPUT_DIR, YOLO_SAM_OUTPUT_DIR, SAM2_CONFIG_FILE, SAM2_CHECKPOINT

# API_KEY = os.environ.get("RECALL_API_KEY") # Original API key variable name
API_KEY = os.environ.get("ROBOFLOW_API_KEY") # Changed to ROBOFLOW_API_KEY to match docker command
# MODEL_ID = "eye-tracking-7udhe" # Original Model ID
MODEL_ID = "infant-eye-tracking-toy2"      # Updated Model ID from user's Roboflow information
MODEL_VERSION = "5"                 # Updated Model Version from user's Roboflow information
INFERENCE_SERVER_URL = "http://localhost:9001/infer/object_detection" # Assuming local docker deployment URL is correct

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_CONFIDENCE_THRESHOLD = 0.5
MAX_WORKERS = 16
BATCH_SIZE = 16

##############################################################################
#                                  DATASET                                   #
##############################################################################

class FrameDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        frame = cv2.imread(str(img_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        return {
            'path': img_path,
            'frame': frame_tensor.pin_memory() if torch.cuda.is_available() else frame_tensor
        }

##############################################################################
#                              MAIN LOGIC                                    #
##############################################################################

class OptimizedYOLOSAM2Runner:
    def __init__(
        self,
        input_frames_dir,
        output_base_dir,
        api_key=API_KEY,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        inference_url=INFERENCE_SERVER_URL,
        confidence_thresh=YOLO_CONFIDENCE_THRESHOLD,
        sam2_config_file=SAM2_CONFIG_FILE,
        sam2_checkpoint_file=SAM2_CHECKPOINT,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS
    ):
        self.input_frames_dir = Path(input_frames_dir)
        self.output_base_dir = Path(output_base_dir)
        self.api_key = api_key
        self.model_id = model_id
        self.model_version = model_version
        self.inference_url = inference_url
        self.confidence_thresh = confidence_thresh
        self.sam2_config_file = sam2_config_file
        self.sam2_checkpoint_file = sam2_checkpoint_file
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.set_per_process_memory_fraction(0.85)
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            torch.cuda.empty_cache()

        self.streams = [torch.cuda.Stream() for _ in range(3)] if torch.cuda.is_available() else []
        self.sam2_lock = Lock()
        self.session = requests.Session()
        retries = requests.packages.urllib3.util.retry.Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(
            max_retries=retries,
            pool_connections=50,
            pool_maxsize=50
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.output_dir = self.create_output_dirs()
        self.json_dir = self.output_dir / "detections"
        self.visual_bbox_dir = self.output_dir / "visual_outputs" / "with_bbox"
        self.visual_nobbox_dir = self.output_dir / "visual_outputs" / "without_bbox"
        self.directory_status_info = ""
        self.font = self.load_font()
        self.sam2_predictor = self.load_sam2_model()
        self.class_color_map = {}
        self.pastel_colors = [
            (255, 204, 255),  # Light Pink
            (204, 255, 204),  # Light Green
            (204, 204, 255),  # Light Lavender
            (255, 240, 204),  # Light Yellow
            (204, 240, 255),  # Light Blue
            (240, 204, 255),  # Light Mauve
            (255, 204, 204),  # Light Peach
            (224, 224, 224)   # Light Gray
        ]
        self.special_colors = {
            "blue_dot":    (203, 192, 255), # Light Purple-Blue (from original)
            "man_body":    (230, 216, 173), # Pale Yellow-Brown (from original)
            "man_hands":   (0, 165, 255),   # Orange (from original, might want to change to pastel)
            "man_face":    (0, 140, 255),    # Orange-Red (from original, might want to change to pastel)
            "woman_body":  (97, 105, 255),   # Light Purple (from original)
            "woman_face":  (0, 128, 0),     # Green (from original, might want to change to pastel)
            "woman_hands": (0, 255, 255),   # Cyan (from original, might want to change to pastel)
            "toy":         (128, 0, 128),    # Purple (from original, might want to change to pastel)
            "toy2":        (255, 255, 224)   # Light Yellow-Beige (Kawaii Pastel for toy2)
        }
        self.pastel_index = 0 # Initialize pastel color index

    def create_output_dirs(self):
        final_output_dir = self.output_base_dir / "inference_output"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        (final_output_dir / "detections").mkdir(parents=True, exist_ok=True)
        visual_output_dir = final_output_dir / "visual_outputs"
        visual_output_dir.mkdir(parents=True, exist_ok=True)
        (visual_output_dir / "with_bbox").mkdir(parents=True, exist_ok=True)
        (visual_output_dir / "without_bbox").mkdir(parents=True, exist_ok=True)
        return final_output_dir

    def load_font(self, font_path=None, size=15):
        try:
            if font_path and Path(font_path).exists():
                return ImageFont.truetype(font_path, size)
            else:
                return ImageFont.load_default()
        except Exception as e:
            print(f"Error loading font: {e}. Using default font.")
            return ImageFont.load_default()

    def load_sam2_model(self):
        try:
            sam2_model = build_sam2(self.sam2_config_file, self.sam2_checkpoint_file, device=self.device)
            predictor = SAM2ImagePredictor(sam2_model, device=self.device)
            print("SAM2.1 model is loaded.")
            return predictor
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            return None

    def get_class_color(self, class_name):
        if class_name in self.special_colors:
            color = self.special_colors[class_name]
        elif class_name not in self.class_color_map:
            color = self.pastel_colors[self.pastel_index % len(self.pastel_colors)] # Cycle through pastel colors
            self.class_color_map[class_name] = color
            self.pastel_index += 1 # Move to the next pastel color for the next new class
        else:
            color = self.class_color_map[class_name] # Use existing color if class already seen
        return color

    def draw_yolo_boxes(self, image, predictions):
        annotated_image = image.copy()
        for pred in predictions:
            class_name = pred.get("class", "N/A")
            confidence = pred.get("confidence", 0)
            x_center   = pred.get("x", 0)
            y_center   = pred.get("y", 0)
            bbox_w     = pred.get("width", 0)
            bbox_h     = pred.get("height", 0)
            x0 = int(x_center - bbox_w / 2)
            y0 = int(y_center - bbox_h / 2)
            x1 = int(x_center + bbox_w / 2)
            y1 = int(y_center + bbox_h / 2)
            padding = 10
            x0 = max(x0 + padding, 0)
            y0 = max(y0 + padding, 0)
            x1 = min(x1 - padding, image.shape[1] - 1)
            y1 = min(y1 - padding, image.shape[0] - 1)
            color_bgr = self.get_class_color(class_name)
            cv2.rectangle(annotated_image, (x0, y0), (x1, y1), color_bgr, 2)
            cv2.putText(annotated_image, f"{class_name} {confidence:.2f}",
                        (x0, max(y0 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        return annotated_image

    def apply_sam_segmentation(self, image, predictions):
        annotated_image = image.copy()
        final_predictions_sam = []
        for pred in predictions: # Iterate through YOLO predictions to use as prompts for SAM
            class_name = pred.get("class", "N/A")
            confidence = pred.get("confidence", 0)
            x_center   = pred.get("x", 0)
            y_center   = pred.get("y", 0)
            bbox_w     = pred.get("width", 0)
            bbox_h     = pred.get("height", 0)

            x0 = int(x_center - bbox_w / 2)
            y0 = int(y_center - bbox_h / 2)
            x1 = int(x_center + bbox_w / 2)
            y1 = int(y_center + bbox_h / 2)

            # Refine bounding box by shrinking (optional) - using same padding as YOLO bbox
            padding = 10
            x0 = max(x0 + padding, 0)
            y0 = max(y0 + padding, 0)
            x1 = min(x1 - padding, image.shape[1] - 1) # Use frame_bgr_sam shape
            y1 = min(y1 - padding, image.shape[0] - 1) # Use frame_bgr_sam shape

            if x1 <= x0 or y1 <= y0 or self.sam2_predictor is None:
                final_predictions_sam.append({ # Or append to final_predictions
                    "class": class_name,
                    "confidence": confidence,
                    "mask_pixels": []
                })
                continue

            try:
                input_point = np.array([[x_center, y_center]])
                input_label = np.array([1])

                with self.sam2_lock:
                    self.sam2_predictor.set_image(image) # Use the re-read image
                    with torch.inference_mode():
                        with torch.cuda.amp.autocast():
                            masks, scores, _ = self.sam2_predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                box=[x0, y0, x1, y1],
                                multimask_output=True
                            )

                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()

                if len(masks) > 0:
                    best_mask_index = np.argmax(scores)
                    segmask = masks[best_mask_index].astype(np.uint8)

                    binary_mask = (segmask > 0).astype(np.uint8) * 255
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

                    refined_mask = cv2.GaussianBlur(refined_mask, (3, 3), 0)
                    refined_mask = (refined_mask > 127).astype(np.uint8)

                    mask_positions = np.argwhere(refined_mask > 0)
                    mask_positions = mask_positions.tolist()

                    final_predictions_sam.append({ # Or append to final_predictions
                        "class": class_name,
                        "confidence": confidence,
                        "mask_pixels": mask_positions
                    })

                    color_bgr = self.get_class_color(class_name)
                    obj_overlay = annotated_image.copy() # Use annotated_overlay_sam and frame_bgr_sam
                    obj_overlay[refined_mask > 0] = color_bgr
                    alpha = 0.4
                    annotated_image = cv2.addWeighted(obj_overlay, alpha, annotated_image, 1 - alpha, 0) # Update annotated_overlay_sam
                else:
                    final_predictions_sam.append({ # Or append to final_predictions
                        "class": class_name,
                        "confidence": confidence,
                        "mask_pixels": []
                    })
            except Exception as e:
                print(f"SAM2 error for {image_path}: {e}")
                final_predictions_sam.append({ # Or append to final_predictions
                    "class": class_name,
                    "confidence": confidence,
                    "mask_pixels": []
                })
        return annotated_image, final_predictions_sam


    def run_inference_on_image(self, image_path):
        start_local = time.time()
        json_path = self.json_dir / f"{Path(image_path).stem}_detections.json"
        annotated_bbox_save_path = self.visual_bbox_dir / f"{Path(image_path).stem}_annotated.jpg"
        annotated_nobbox_save_path = self.visual_nobbox_dir / f"{Path(image_path).stem}_annotated.jpg"

        if annotated_bbox_save_path.exists() and annotated_nobbox_save_path.exists():
            duration = time.time() - start_local
            return True, f"Skipping {image_path}, already processed.", duration

        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            b64_image = base64.b64encode(img_bytes).decode('utf-8')
            payload = {
                "api_key": self.api_key,
                "model_id": f"{self.model_id}/{self.model_version}",
                "model_type": "object-detection",
                "image": [{"type": "base64", "value": b64_image}],
                "confidence": self.confidence_thresh,
                "iou_threshold": 0.5,
                "max_detections": 300
            }
            resp = self.session.post(self.inference_url, json=payload, timeout=120)
            if resp.status_code != 200:
                duration = time.time() - start_local
                return False, f"Error with Roboflow detection: {resp.status_code} - {resp.text}", duration # Changed error message to Roboflow
            result = resp.json()
            if not isinstance(result, list) or len(result) == 0:
                duration = time.time() - start_local
                return False, "No Roboflow result returned.", duration # Changed error message to Roboflow
            predictions = result[0].get("predictions", [])
        except Exception as e:
            duration = time.time() - start_local
            return False, f"Exception in Roboflow step: {e}", duration # Changed error message to Roboflow

        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            duration = time.time() - start_local
            return False, f"Unable to load {image_path}", duration

        final_predictions_yolo = []
        for pred in predictions:
            class_name = pred.get("class", "N/A")
            confidence = pred.get("confidence", 0)
            x_center   = pred.get("x", 0)
            y_center   = pred.get("y", 0)
            bbox_w     = pred.get("width", 0)
            bbox_h     = pred.get("height", 0)
            final_predictions_yolo.append({
                "class": class_name,
                "confidence": confidence,
                "x_center": x_center,
                "y_center": y_center,
                "width": bbox_w,
                "height": bbox_h
            })

        annotated_overlay_with_bbox = frame_bgr.copy()
        annotated_overlay_with_bbox = self.draw_yolo_boxes(annotated_overlay_with_bbox, predictions)
        annotated_overlay_with_bbox, final_predictions_sam = self.apply_sam_segmentation(annotated_overlay_with_bbox, predictions)

        annotated_overlay_without_bbox = frame_bgr.copy() # Start from the original frame for no bbox version
        annotated_overlay_without_bbox, _ = self.apply_sam_segmentation(annotated_overlay_without_bbox, predictions) # Apply only SAM

        final_predictions = final_predictions_yolo + final_predictions_sam # Combine YOLO and SAM predictions in JSON if needed

        ##################
        # 3) SAVE OUTPUT #
        ##################
        try:
            with open(json_path, "w") as jf:
                json.dump(final_predictions, jf, indent=4)
        except Exception as e:
            print(f"Error saving JSON for {image_path}: {e}")
        try:
            annotated_pil_bbox = Image.fromarray(cv2.cvtColor(annotated_overlay_with_bbox, cv2.COLOR_BGR2RGB))
            annotated_pil_bbox.save(str(annotated_bbox_save_path), quality=95)

            annotated_pil_nobbox = Image.fromarray(cv2.cvtColor(annotated_overlay_without_bbox, cv2.COLOR_BGR2RGB))
            annotated_pil_nobbox.save(str(annotated_nobbox_save_path), quality=95)

        except Exception as e:
            print(f"Error saving annotated image for {image_path}: {e}")
        duration = time.time() - start_local
        return True, str(image_path), duration

    def process_images(self):
        all_files = sorted([p for p in self.input_frames_dir.glob("*.*")
                             if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]])
        total_frames = len(all_files)
        if total_frames == 0:
            self.directory_status_info = f"No frames found in {self.input_frames_dir}"
            print(self.directory_status_info)
            return
        annotated_bbox_imgs = list(self.visual_bbox_dir.glob("*_annotated.*"))
        annotated_nobbox_imgs = list(self.visual_nobbox_dir.glob("*_annotated.*"))
        done_count = min(len(annotated_bbox_imgs), len(annotated_nobbox_imgs)) # Consider done if both exist

        if done_count >= total_frames:
            self.directory_status_info = (f"Skipping {self.input_frames_dir}, all {total_frames} frames appear done.")
            print(self.directory_status_info)
            return
        self.directory_status_info = (f"Processing {self.input_frames_dir} because {done_count}/{total_frames} frames are annotated so far.")
        print(self.directory_status_info)
        print(f"Total frames in directory: {total_frames}")
        print(f"Already done (annotated): {done_count}, not done: {total_frames - done_count}")
        unprocessed_files = []
        for img in all_files:
            annotated_bbox_img = self.visual_bbox_dir / f"{img.stem}_annotated{img.suffix}"
            annotated_nobbox_img = self.visual_nobbox_dir / f"{img.stem}_annotated{img.suffix}"
            if not annotated_bbox_img.exists() or not annotated_nobbox_img.exists(): # Check if BOTH versions exist
                unprocessed_files.append(img)
        if not unprocessed_files:
            print(f"All frames in {self.input_frames_dir} are done, skipping.")
            return
        print(f"Will process {len(unprocessed_files)} unprocessed frames.")
        last_messages = deque(maxlen=10)
        start_time = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_image = {executor.submit(self.run_inference_on_image, img): img for img in unprocessed_files}
            processed_count = 0
            for future in as_completed(future_to_image):
                img_path = future_to_image[future]
                processed_count += 1
                elapsed_total = time.time() - start_time
                rate = processed_count / elapsed_total if elapsed_total > 0 else 0
                est_time_left = (len(unprocessed_files) - processed_count) / rate if rate > 0 else 0
                total_est_time = elapsed_total + est_time_left
                elapsed_min = int(elapsed_total / 60)
                total_est_min = int(total_est_time / 60)
                est_time_left_min = int(est_time_left_min / 60)
                try:
                    result_tuple = future.result()
                    if len(result_tuple) == 3:
                        success, msg, file_duration = result_tuple
                    else:
                        success, msg = result_tuple
                        file_duration = 0.0
                    short_info = f"✓ {Path(msg).name}" if success else f"✗ {msg}"
                except Exception as e:
                    success = False
                    msg = f"Exception: {img_path} -> {e}"
                    file_duration = 0.0
                    short_info = f"✗ {msg}"
                line_message = (f"[{processed_count}/{len(unprocessed_files)}] "
                                f"[{rate:.2f} it/s] Rate: {rate:.2f} imgs/s, "
                                f"took {file_duration:.2f}s -- {short_info}")
                last_messages.append(line_message)
                results.append((success, msg))
                os.system('cls' if os.name == 'nt' else 'clear')
                print(self.directory_status_info)
                print(f"Elapsed time: {elapsed_min} min")
                print(f"Estimated total time: {total_est_min} min (about {est_time_left_min} min remaining)")
                print(f"Directory: {self.input_frames_dir}")
                percent_done = (processed_count / len(unprocessed_files)) * 100
                print(f"Progress: {percent_done:0.1f}% ({processed_count}/{len(unprocessed_files)})")
                print("\n--- Last 10 processed frames ---")
                for msg_line in last_messages:
                    print(msg_line)
        successes = sum(1 for r in results if r[0])
        print(f"\nDone. Successfully processed {successes}/{len(results)} frames.")
        print(f"Results are in {self.output_base_dir}")

    def monitor_performance(self):
        if torch.cuda.is_available():
            print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"Max GPU Memory Usage: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            try:
                gpu_util = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
                ).decode("utf-8").strip()
                print(f"GPU Utilization: {gpu_util}%")
            except Exception as e:
                print(f"Failed to get GPU utilization: {e}")

def main():
    base_input_dir = Path(YOLO_SAM_INPUT_DIR)  # Note: For the SAM script this will be changed below.
    # This main() function will not be used in the SAM script.
    # The YOLO+SAM script main is defined separately.
    output_base_dir = Path(YOLO_SAM_OUTPUT_DIR)
    if not base_input_dir.exists():
        print(f"Input directory does not exist: {base_input_dir}")
        return
    output_base_dir.mkdir(parents=True, exist_ok=True)
    frames_dirs = [d for d in base_input_dir.iterdir() if d.is_dir() and d.name.endswith("-1024-frames")]
    if not frames_dirs:
        print(f"No participant directories found in {YOLO_SAM_INPUT_DIR} with '-1024-frames' in the name.")
        return
    print(f"Found {len(frames_dirs)} participant directories to process.")
    for frames_dir in frames_dirs:
        participant_name = frames_dir.name
        participant_output_dir = output_base_dir / participant_name
        participant_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing participant directory: {frames_dir}")
        runner = OptimizedYOLOSAM2Runner(
            input_frames_dir=frames_dir,
            output_base_dir=participant_output_dir,
            sam2_config_file=SAM2_CONFIG_FILE, # Pass SAM2 config path from config.py
            sam2_checkpoint_file=SAM2_CHECKPOINT # Pass SAM2 checkpoint path from config.py
        )
        runner.process_images()
        runner.monitor_performance()
if __name__ == "__main__":
    main()