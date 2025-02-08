#!/usr/bin/env python3
# 2_inference_object_yolo.py

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
from torch.utils.data import DataLoader, Dataset
from collections import deque

import requests.packages.urllib3.util.retry
import requests.adapters

from dotenv import load_dotenv
load_dotenv()

##############################################################################
#                                  SETTINGS                                  #
##############################################################################

# For this script, the input comes from preprocessed frames and the output goes to YOLO object detection results.
from config import YOLO_OBJ_INPUT_DIR, YOLO_OBJ_OUTPUT_DIR

API_KEY = os.environ.get("SUBJECTS_API_KEY")
MODEL_ID = "infant-eye-tracking-toy2"
MODEL_VERSION = "5"
INFERENCE_SERVER_URL = "http://localhost:9001/infer/object_detection"

YOLO_CONFIDENCE_THRESHOLD = 0.50
MAX_WORKERS = 16
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

class OptimizedYOLORunner:
    def __init__(
        self,
        input_frames_dir,
        output_base_dir,
        api_key=API_KEY,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        inference_url=INFERENCE_SERVER_URL,
        confidence_thresh=YOLO_CONFIDENCE_THRESHOLD,
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

        self.session = requests.Session()
        retries = requests.packages.urllib3.util.retry.Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.json_dir, self.visual_dir = self.create_output_dirs()
        self.font = self.load_font()
        self.class_color_map = {}
        self.special_colors = {
            "blue_dot":    (203, 192, 255),
            "man_body":    (230, 216, 173),
            "man_hands":   (0, 165, 255),
            "man_face":    (0, 140, 255),
            "woman_body":  (97, 105, 255),
            "woman_face":  (0, 128, 0),
            "woman_hands": (0, 255, 255),
            "toy":         (128, 0, 128),
            "toy2":        (235, 206, 135)
        }

    def create_output_dirs(self):
        participant_name = self.output_base_dir.name
        json_dir = self.output_base_dir / f"detections-{participant_name}"
        visual_dir = self.output_base_dir / f"image_segmentation-{participant_name}"
        json_dir.mkdir(parents=True, exist_ok=True)
        visual_dir.mkdir(parents=True, exist_ok=True)
        return json_dir, visual_dir

    def load_font(self, font_path=None, size=15):
        try:
            if font_path and Path(font_path).exists():
                return ImageFont.truetype(font_path, size)
            else:
                return ImageFont.load_default()
        except Exception as e:
            print(f"Error loading font: {e}. Using default font.")
            return ImageFont.load_default()

    def get_class_color(self, class_name):
        if class_name in self.special_colors:
            self.class_color_map[class_name] = self.special_colors[class_name]
        elif class_name not in self.class_color_map:
            color = np.random.randint(120, 201, size=3, dtype=np.uint8)
            self.class_color_map[class_name] = tuple(int(x) for x in color)
        return self.class_color_map[class_name]

    def run_inference_on_image(self, image_path):
        start_local = time.time()
        json_path = self.json_dir / f"{Path(image_path).stem}_detections.json"
        annotated_save_path = self.visual_dir / f"{Path(image_path).stem}_annotated.jpg"

        if json_path.exists() and annotated_save_path.exists():
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
                return False, f"Error with YOLO detection: {resp.status_code} - {resp.text}", duration
            result = resp.json()
            if not isinstance(result, list) or len(result) == 0:
                duration = time.time() - start_local
                return False, "No YOLO result returned.", duration
            predictions = result[0].get("predictions", [])
        except Exception as e:
            duration = time.time() - start_local
            return False, f"Exception in YOLO step: {e}", duration

        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            duration = time.time() - start_local
            return False, f"Unable to load {image_path}", duration

        annotated_overlay = frame_bgr.copy()
        final_predictions = []
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
            x1 = min(x1 - padding, frame_bgr.shape[1] - 1)
            y1 = min(y1 - padding, frame_bgr.shape[0] - 1)
            final_predictions.append({
                "class": class_name,
                "confidence": confidence,
                "x_center": x_center,
                "y_center": y_center,
                "width": bbox_w,
                "height": bbox_h
            })
            color_bgr = self.get_class_color(class_name)
            cv2.rectangle(annotated_overlay, (x0, y0), (x1, y1), color_bgr, 2)
            cv2.putText(annotated_overlay, f"{class_name} {confidence:.2f}",
                        (x0, max(y0 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        try:
            with open(json_path, "w") as jf:
                json.dump(final_predictions, jf, indent=4)
        except Exception as e:
            print(f"Error saving JSON for {image_path}: {e}")
        try:
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_overlay, cv2.COLOR_BGR2RGB))
            annotated_pil.save(str(annotated_save_path), quality=95)
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

        annotated_imgs = list(self.visual_dir.glob("*_annotated.*"))
        done_count = len(annotated_imgs)
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
            annotated_img = self.visual_dir / f"{img.stem}_annotated{img.suffix}"
            if not annotated_img.exists():
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
                est_time_left_min = int(est_time_left / 60)
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
                                f"[{rate:.2f} it/s] Rate: {rate:.2f} imgs/s, took {file_duration:.2f}s -- {short_info}")
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
    base_input_dir = Path(YOLO_OBJ_INPUT_DIR)
    output_base_dir = Path(YOLO_OBJ_OUTPUT_DIR)
    if not base_input_dir.exists():
        print(f"Input directory does not exist: {base_input_dir}")
        return
    output_base_dir.mkdir(parents=True, exist_ok=True)
    frames_dirs = [d for d in base_input_dir.iterdir() if d.is_dir() and d.name.endswith("-1024-frames")]
    if not frames_dirs:
        print(f"No participant directories found in {YOLO_OBJ_INPUT_DIR} with '-1024-frames' in the name.")
        return
    print(f"Found {len(frames_dirs)} directories to process.")
    for frames_dir in frames_dirs:
        segmentation_name = frames_dir.name.replace("-frames", "-segmentation")
        inference_output_dir = output_base_dir / segmentation_name
        inference_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing frames directory: {frames_dir}")
        runner = OptimizedYOLORunner(
            input_frames_dir=frames_dir,
            output_base_dir=inference_output_dir
        )
        runner.process_images()
        runner.monitor_performance()
if __name__ == "__main__":
    main()