#!/usr/bin/env python3
# 4_inference_background.py

import os
import time
from pathlib import Path
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import base64
import glob

class EventRecognitionRunner:
    def __init__(self, input_frames_dir, output_base_dir, participant_name):
        """
        input_frames_dir: Path to the frames directory (e.g. 'FiftySix-0501-1673-1024-frames').
        output_base_dir: Global output directory for detection results.
        participant_name: The name of the participant folder (e.g., 'FiftySix-0501-1673-1024-frames').
        """
        self.input_frames_dir = Path(input_frames_dir)
        self.output_base_dir = Path(output_base_dir)
        self.participant_name = participant_name
        
        # Inference Server Configuration
        self.inference_server_url = "http://localhost:9001/infer/classification"
        self.api_key = "e9GxVRpbrzHTlbRBnvMC"
        self.model_id = "events-recognition/2"
        self.model_type = "classification"
        
        # Create (or resume) the participant's output directory
        # Change the folder name from "*-frames" to "*-inference"
        self.classification_dir = self._create_or_resume_output_dir()
        
        # Inside that, create a subfolder for detections
        self.detections_dir = self.classification_dir / "detections"
        self.detections_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV path is stored in the participant's output folder
        self.csv_output_path = self.classification_dir / "detections_summary.csv"
        
        # A place to hold CSV data during processing
        self.csv_data = self._load_existing_csv_entries()
        
        # Classes (for reference)
        self.classes = [
            "green_dot", "f", "gw", "gwo", "hw", "hwo",
            "sw", "swo", "ugw", "ugwo", "uhw", "uhwo"
        ]
        
        # HTTP Session with retries
        self.session = requests.Session()
        retries = requests.packages.urllib3.util.retry.Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        print(f"Detections output directory: {self.classification_dir}")
        print(f"- Detections will be saved in: {self.detections_dir}")
        print(f"- CSV summary will be saved at: {self.csv_output_path}")

    def _create_or_resume_output_dir(self):
        """
        Instead of using a classification-run-* structure,
        change the participant folder name from "*-frames" to "*-inference".
        """
        desired_name = self.participant_name.replace("frames", "inference")
        classification_run_dir = self.output_base_dir / desired_name
        classification_run_dir.mkdir(parents=True, exist_ok=True)
        return classification_run_dir

    def _load_existing_csv_entries(self):
        data = []
        if self.csv_output_path.exists():
            try:
                with open(self.csv_output_path, mode='r', encoding='utf-8') as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        frame_num = row.get("frame_number", "")
                        events = row.get("events", "")
                        try:
                            frame_num = int(frame_num)
                        except ValueError:
                            pass
                        data.append({"frame_number": frame_num, "events": events})
                print(f"Resuming from existing CSV: {self.csv_output_path}")
            except Exception as e:
                print(f"Error reading existing CSV ({self.csv_output_path}): {e}")
        return data

    def _is_frame_already_processed(self, frame_path):
        json_path = self.detections_dir / f"{frame_path.stem}_detections.json"
        if json_path.exists():
            return True
        
        frame_number = self._get_frame_number_from_name(frame_path.stem)
        if any(entry["frame_number"] == frame_number for entry in self.csv_data):
            return True
        
        return False

    def _get_frame_number_from_name(self, frame_stem):
        if frame_stem.startswith("frame_"):
            num_part = frame_stem.replace("frame_", "")
            try:
                return int(num_part)
            except ValueError:
                return frame_stem
        return frame_stem
    
    def _save_detections(self, frame_path, predictions):
        json_path = self.detections_dir / f"{frame_path.stem}_detections.json"
        try:
            with open(json_path, 'w') as f:
                json.dump(predictions, f, indent=4)
        except Exception as e:
            print(f"Error saving JSON for {frame_path.name}: {e}")
        
        events_detected = [p["class"] for p in predictions if p.get("confidence", 0) >= 0.5]
        
        frame_number = self._get_frame_number_from_name(frame_path.stem)
        self.csv_data.append({
            "frame_number": frame_number,
            "events": ";".join(events_detected) if events_detected else "None"
        })

    def get_frame_paths(self):
        frame_extensions = ['*.jpg', '*.jpeg', '*.png']
        frame_paths = []
        for ext in frame_extensions:
            frame_paths.extend(self.input_frames_dir.glob(ext))
        frame_paths = sorted(frame_paths)
        print(f"Found {len(frame_paths)} frame(s) in {self.input_frames_dir}")
        return frame_paths

    def prepare_payload(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            payload = {
                "id": f"inference_{image_path.stem}",
                "api_key": self.api_key,
                "usage_billable": True,
                "start": 0,
                "source": "script",
                "source_info": "Infant Eye Tracking Event Labeling",
                "model_id": self.model_id,
                "model_type": self.model_type,
                "image": [
                    {
                        "type": "base64",
                        "value": image_base64
                    }
                ],
                "confidence": 0.5,
            }
            return payload
        except Exception as e:
            print(f"Error preparing payload for {image_path.name}: {e}")
            return None

    def make_inference_request(self, payload):
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            response = self.session.post(
                self.inference_server_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=120
            )
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Response content: {response.text}")
                    return None
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return None

    def process_single_frame(self, frame_path):
        if self._is_frame_already_processed(frame_path):
            info = f"Skipping {frame_path.name} (already processed)."
            self._add_line(info)
            return False, frame_path
        
        payload = self.prepare_payload(frame_path)
        if payload is None:
            info = f"Payload error for {frame_path.name} (skipped)."
            self._add_line(info)
            return False, frame_path
        
        response_json = self.make_inference_request(payload)
        if response_json is None:
            info = f"Server error for {frame_path.name} (skipped)."
            self._add_line(info)
            return False, frame_path
        
        if isinstance(response_json, list):
            if not response_json:
                predictions = []
            else:
                response_json = response_json[0]
                predictions = response_json.get("predictions", [])
        elif isinstance(response_json, dict):
            predictions = response_json.get("predictions", [])
        else:
            predictions = []
        
        self._save_detections(frame_path, predictions)
        
        events_detected = [p["class"] for p in predictions if p.get("confidence", 0) >= 0.5]
        if events_detected:
            info = f"Processed {frame_path.name} - Detected: {', '.join(events_detected)}"
        else:
            info = f"Processed {frame_path.name} - No events found"
        
        self._add_line(info)
        return True, frame_path

    def save_csv(self):
        try:
            with open(self.csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['frame_number', 'events']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                
                def sort_key(x):
                    return x['frame_number'] if isinstance(x['frame_number'], int) else str(x['frame_number'])
                
                for row in sorted(self.csv_data, key=sort_key):
                    writer.writerow(row)
        except Exception as e:
            print(f"Error saving CSV: {e}")
        else:
            print(f"CSV summary updated: {self.csv_output_path}")

    def _init_console_tracker(self, total_frames):
        self.start_time = time.time()
        self.frames_processed = 0
        self.total_frames_to_process = total_frames
        self.last_10_lines = []

    def _add_line(self, text):
        self.last_10_lines.append(text)
        if len(self.last_10_lines) > 10:
            self.last_10_lines.pop(0)

    def _update_console_status(self):
        elapsed = time.time() - self.start_time
        fps = self.frames_processed / elapsed if elapsed > 0 else 0
        remaining_frames = self.total_frames_to_process - self.frames_processed
        remaining_time = remaining_frames / fps if fps > 0 else 0

        def format_time(seconds):
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h}h {m}m {s}s"
            elif m > 0:
                return f"{m}m {s}s"
            else:
                return f"{s}s"
        
        print("\033c", end="")
        print(f"Processing directory: {self.input_frames_dir}")
        print(f"Elapsed time: {format_time(elapsed)} | ETA: {format_time(remaining_time)}")
        print(f"Processed frames: {self.frames_processed}/{self.total_frames_to_process}")
        print("--------------------------------------------------")
        print("Last 10 lines:")
        for line in self.last_10_lines:
            print(line)

    def run_inference(self):
        frame_paths = self.get_frame_paths()
        if not frame_paths:
            print(f"No frames found under {self.input_frames_dir}. Nothing to do.")
            return
        
        self._init_console_tracker(total_frames=len(frame_paths))

        print("\nStarting classification on frames...")

        max_workers = 4
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_frame = {executor.submit(self.process_single_frame, fp): fp for fp in frame_paths}
                for future in tqdm(as_completed(future_to_frame), total=len(future_to_frame), desc="Processing frames"):
                    fp = future_to_frame[future]
                    try:
                        success, frame_path = future.result()
                    except Exception as exc:
                        self._add_line(f"Exception processing {fp.name}: {exc}")
                    self.frames_processed += 1
                    self._update_console_status()
        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down gracefully...")
            executor.shutdown(wait=False)
            return
        
        self.save_csv()
        
        processed_count = sum(1 for x in self.csv_data if x["events"] != "None")
        print("\nClassification completed!")
        print(f"Out of {len(frame_paths)} frame(s) total, {len(self.csv_data)} have CSV entries.")
        print(f"Of those, {processed_count} had at least one detected event.")
        print(f"Results saved in: {self.classification_dir}")


def main():
    """
    1. Find all participant folders (each ending in '-1024-frames') under
       D:\infant eye-tracking\paper-area\1_preprocessing_output.
    2. For each, create (or resume) a detection output folder in the global output folder
       D:\infant eye-tracking\paper-area\4_inference_background_output, with the participant's
       name modified from '*-frames' to '*-inference'. Inside that folder, a subfolder named
       'detections' is created to store detection JSON files, and the CSV summary is stored in
       the participant folder.
    3. Classify all frames inside, skipping already-processed frames.
    """
    root_dir = Path(r"D:\infant eye-tracking\paper-area\1_preprocessing_output")
    extracted_frames_dirs = list(root_dir.glob("*-1024-frames"))
    if not extracted_frames_dirs:
        print(f"No '-1024-frames' directories found under {root_dir}. Nothing to classify.")
        return

    print(f"Found {len(extracted_frames_dirs)} participant folder(s).")
    
    global_output_dir = Path(r"D:\infant eye-tracking\paper-area\4_inference_background_output")
    global_output_dir.mkdir(parents=True, exist_ok=True)
    
    dir_counter = 0
    for frames_dir in extracted_frames_dirs:
        dir_counter += 1
        participant_name = frames_dir.stem  # e.g., "FiftySix-0501-1673-1024-frames"
        
        print("\n==========================================================")
        print(f"Starting classification for participant #{dir_counter}:")
        print(f"Frames folder: {frames_dir}")
        print(f"Global output folder: {global_output_dir}")

        runner = EventRecognitionRunner(
            input_frames_dir=frames_dir,
            output_base_dir=global_output_dir,
            participant_name=participant_name
        )
        
        print("\n=== Classification Docker/Server Setup Instructions ===")
        print("1. Install Docker if you haven't already.")
        print("2. Ensure the Inference Server is running, for example:")
        print(f"   docker run -d --gpus all -p 9001:9001 -e ROBOFLOW_API_KEY={runner.api_key} "
              "-v C:/Users/yurig/roboflow_cache:/tmp/cache roboflow/roboflow-inference-server-gpu:latest")
        print("========================================================\n")
        
        runner.run_inference()

if __name__ == "__main__":
    main()
