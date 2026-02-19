"""
AI Traffic Monitoring System — Main Runner
=============================================
Production-level intelligent traffic monitoring system.

Features:
    - Vehicle detection (YOLOv8)
    - License plate detection + OCR (EasyOCR)
    - Vehicle color detection (HSV analysis)
    - Traffic violation detection (configurable rules)
    - Multi-object tracking (SORT)
    - CSV data logging
    - Real-time visual dashboard

Usage:
    python main.py --source video.mp4
    python main.py --source 0                    # Webcam
    python main.py --source rtsp://camera_url    # RTSP stream
    python main.py --source video.mp4 --no-display --output output_video.mp4
"""

import os
import sys
import ssl
import argparse
import time
import cv2
import numpy as np
from datetime import datetime

# ── Fix SSL certificate verification errors (common on Windows) ──
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.vehicle_detector import VehicleDetector
from modules.plate_detector import PlateDetector
from modules.ocr_reader import OCRReader
from modules.color_detector import ColorDetector
from modules.tracker import VehicleTracker
from modules.violation_detector import ViolationDetector
from modules.csv_logger import CSVLogger
from modules.video_input import VideoInput
from modules.dashboard import Dashboard


class TrafficMonitoringSystem:
    """
    Main orchestrator for the AI Traffic Monitoring System.
    
    Coordinates all modules in a pipeline:
        Video Input → Vehicle Detection → Tracking → Plate Detection →
        OCR → Color Detection → Violation Detection → CSV Logging → Dashboard
    """

    def __init__(self, args):
        """
        Initialize all system components.

        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.running = True

        print("=" * 60)
        print("  AI Traffic Monitoring System")
        print("  Initializing components...")
        print("=" * 60)

        # Initialize modules
        self._init_vehicle_detector()
        self._init_plate_detector()
        self._init_ocr_reader()
        self._init_color_detector()
        self._init_tracker()
        self._init_violation_detector()
        self._init_csv_logger()
        self._init_video_input()
        self._init_dashboard()
        self._init_video_writer()

        # Processing state
        self.processed_tracks = {}  # track_id -> {plate, color, violations}
        self.frame_skip = args.frame_skip

        print("\n[System] All components initialized successfully.")
        print(f"[System] Source: {args.source}")
        print(f"[System] Output CSV: {args.csv_output}")
        print(f"[System] Display: {'Enabled' if not args.no_display else 'Disabled'}")
        print("=" * 60)

    def _init_vehicle_detector(self):
        """Initialize the YOLOv8 vehicle detector."""
        print("[Init] Loading vehicle detection model...")
        self.vehicle_detector = VehicleDetector(
            model_path=self.args.yolo_model,
            confidence_threshold=self.args.confidence,
            device=self.args.device,
        )

    def _init_plate_detector(self):
        """Initialize the license plate detector."""
        print("[Init] Initializing plate detector...")
        self.plate_detector = PlateDetector(
            min_plate_area=self.args.min_plate_area,
        )

    def _init_ocr_reader(self):
        """Initialize the OCR reader."""
        print("[Init] Loading OCR engine (this may take a moment)...")
        self.ocr_reader = OCRReader(
            languages=["en"],
            gpu=(self.args.device != "cpu"),
            min_confidence=0.3,
        )

    def _init_color_detector(self):
        """Initialize the vehicle color detector."""
        print("[Init] Initializing color detector...")
        self.color_detector = ColorDetector()

    def _init_tracker(self):
        """Initialize the multi-object tracker."""
        print("[Init] Initializing vehicle tracker...")
        self.tracker = VehicleTracker(
            max_age=self.args.max_age,
            min_hits=self.args.min_hits,
            iou_threshold=0.3,
        )

    def _init_violation_detector(self):
        """Initialize the violation detector."""
        print("[Init] Initializing violation detector...")
        config_path = self.args.violation_config if os.path.exists(
            self.args.violation_config
        ) else None
        self.violation_detector = ViolationDetector(config_path=config_path)

    def _init_csv_logger(self):
        """Initialize the CSV logger."""
        print("[Init] Initializing CSV logger...")
        self.csv_logger = CSVLogger(output_path=self.args.csv_output)

    def _init_video_input(self):
        """Initialize the video input source."""
        print("[Init] Opening video source...")
        # Determine source type
        source = self.args.source
        try:
            source = int(source)  # Try webcam index
        except ValueError:
            pass  # Keep as string (file path or URL)

        self.video_input = VideoInput(
            source=source,
            target_fps=self.args.target_fps,
            resize=tuple(self.args.resize) if self.args.resize else None,
        )

    def _init_dashboard(self):
        """Initialize the visual dashboard."""
        print("[Init] Initializing dashboard...")
        self.dashboard = Dashboard(
            show_counts=True,
            show_fps=True,
            show_timestamp=True,
        )

    def _init_video_writer(self):
        """Initialize video writer for output recording."""
        self.video_writer = None
        if self.args.output_video:
            print(f"[Init] Video output will be saved to: {self.args.output_video}")

    def run(self):
        """
        Main processing loop.
        Reads frames, processes them through the pipeline, and displays results.
        """
        if not self.video_input.open():
            print("[Error] Failed to open video source. Exiting.")
            return

        info = self.video_input.get_info()
        print(f"\n[Running] Processing {info.get('source_type', 'unknown')} source...")
        print("[Running] Press 'q' to quit, 's' to save snapshot, 'r' to reset counts\n")

        frame_idx = 0

        try:
            for frame in self.video_input.frames():
                if not self.running:
                    break

                frame_idx += 1

                # Frame skipping for performance
                if frame_idx % (self.frame_skip + 1) != 0:
                    continue

                # Process frame through pipeline
                tracked_objects = self._process_frame(frame)

                # Render dashboard
                display_frame = self.dashboard.render(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    vehicle_counts=self.vehicle_detector.get_counts(),
                    total_logged=self.csv_logger.get_entry_count(),
                )

                # Write output video
                if self.args.output_video:
                    self._write_frame(display_frame)

                # Display
                if not self.args.no_display:
                    key = Dashboard.show_frame(
                        "AI Traffic Monitor", display_frame
                    )
                    if key == ord("q"):
                        print("\n[System] Quit requested by user.")
                        break
                    elif key == ord("s"):
                        snapshot_path = os.path.join(
                            "output", f"snapshot_{frame_idx}.jpg"
                        )
                        Dashboard.save_frame(snapshot_path, display_frame)
                        print(f"[System] Snapshot saved: {snapshot_path}")
                    elif key == ord("r"):
                        self.vehicle_detector.reset_counts()
                        print("[System] Vehicle counts reset.")

        except KeyboardInterrupt:
            print("\n[System] Interrupted by user.")
        finally:
            self._cleanup()

    def _process_frame(self, frame: np.ndarray) -> list:
        """
        Process a single frame through the full detection pipeline.

        Args:
            frame: Input BGR frame

        Returns:
            List of tracked objects with all metadata
        """
        # Step 1: Detect vehicles
        detections = self.vehicle_detector.detect(frame)

        # Step 2: Track vehicles
        tracked = self.tracker.update(detections)

        # Step 3: Process each tracked vehicle
        enriched_objects = []

        for obj in tracked:
            track_id = obj["track_id"]
            bbox = obj["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            class_name = obj.get("class_name", "Unknown")
            confidence = obj.get("confidence", 0.0)

            # Calculate center for tracking
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Update violation detector with position
            self.violation_detector.update_track(track_id, (cx, cy))

            # Initialize track data if new
            if track_id not in self.processed_tracks:
                self.processed_tracks[track_id] = {
                    "plate_number": None,
                    "plate_confidence": 0.0,
                    "vehicle_color": None,
                    "violations": [],
                    "logged": False,
                    "attempts": 0,
                }

            track_data = self.processed_tracks[track_id]

            # Crop vehicle region
            vehicle_crop = VehicleDetector.crop_vehicle(frame, (x1, y1, x2, y2))

            if vehicle_crop.size == 0:
                continue

            # Step 3a: Detect vehicle color (once per track)
            if track_data["vehicle_color"] is None:
                track_data["vehicle_color"] = self.color_detector.detect_color(
                    vehicle_crop
                )

            # Step 3b: Detect and read license plate (retry up to N times)
            if (
                track_data["plate_number"] is None
                and track_data["attempts"] < self.args.max_ocr_attempts
            ):
                track_data["attempts"] += 1
                plate_result = self.plate_detector.detect_plate(vehicle_crop)

                if plate_result is not None:
                    plate_crop, plate_bbox = plate_result
                    processed_plate = PlateDetector.preprocess_plate(plate_crop)

                    plate_text, plate_conf = self.ocr_reader.read_plate(
                        processed_plate
                    )

                    if plate_text and plate_conf > track_data["plate_confidence"]:
                        track_data["plate_number"] = plate_text
                        track_data["plate_confidence"] = plate_conf

                        # Save plate crop
                        if self.args.save_plates:
                            plate_path = os.path.join(
                                "output", "plates",
                                f"plate_{track_id}_{plate_text}.jpg",
                            )
                            cv2.imwrite(plate_path, plate_crop)

            # Step 3c: Check for violations
            violations = self.violation_detector.detect_violations(
                track_id=track_id,
                vehicle_type=class_name,
                bbox=(x1, y1, x2, y2),
                frame_shape=frame.shape[:2],
            )
            if violations:
                track_data["violations"] = list(
                    set(track_data["violations"] + violations)
                )

            # Step 3d: Log to CSV (once per track, when plate is available)
            if (
                not track_data["logged"]
                and track_data["plate_number"] is not None
                and obj.get("is_new", False)
            ):
                self.csv_logger.log_violation(
                    plate_number=track_data["plate_number"],
                    vehicle_type=class_name,
                    vehicle_color=track_data["vehicle_color"] or "Unknown",
                    violations=track_data["violations"] if track_data["violations"] else ["None"],
                    confidence=track_data["plate_confidence"],
                    track_id=track_id,
                )
                track_data["logged"] = True

                # Update vehicle counts for new tracked vehicles
                self.vehicle_detector.update_counts([obj])

            # Build enriched object for dashboard
            enriched_objects.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "class_name": class_name,
                "confidence": confidence,
                "plate_number": track_data["plate_number"] or "",
                "vehicle_color": track_data["vehicle_color"] or "",
                "violations": track_data["violations"],
            })

        return enriched_objects

    def _write_frame(self, frame: np.ndarray):
        """Write frame to output video file."""
        if self.video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.args.target_fps or 30
            self.video_writer = cv2.VideoWriter(
                self.args.output_video, fourcc, fps, (w, h)
            )
        self.video_writer.write(frame)

    def _cleanup(self):
        """Release all resources."""
        print("\n[System] Cleaning up...")
        self.video_input.release()
        if self.video_writer is not None:
            self.video_writer.release()
        if not self.args.no_display:
            Dashboard.destroy_all()

        # Print summary
        counts = self.vehicle_detector.get_counts()
        total = sum(counts.values())
        logged = self.csv_logger.get_entry_count()

        print("\n" + "=" * 60)
        print("  Session Summary")
        print("=" * 60)
        print(f"  Total vehicles detected: {total}")
        for vtype, count in counts.items():
            print(f"    {vtype}: {count}")
        print(f"  Total entries logged: {logged}")
        print(f"  CSV output: {self.args.csv_output}")
        if self.args.output_video:
            print(f"  Video output: {self.args.output_video}")
        print("=" * 60)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Traffic Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source video.mp4
  python main.py --source 0
  python main.py --source rtsp://192.168.1.100:554/stream
  python main.py --source video.mp4 --no-display --output result.mp4
  python main.py --source video.mp4 --device cuda --confidence 0.5
        """,
    )

    # Input/Output
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: file path, webcam index, or RTSP URL (default: 0)",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="output/detection_log.csv",
        help="Path to CSV output file (default: output/detection_log.csv)",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Path to save output video (optional)",
    )

    # Model settings
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model path or name (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: cpu, cuda, 0, 1, etc. (default: cpu)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold (default: 0.4)",
    )

    # Tracking settings
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Max frames to keep lost tracks (default: 30)",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Min hits before track is confirmed (default: 3)",
    )

    # Processing settings
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Number of frames to skip between processing (default: 0)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Target processing FPS (default: source FPS)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        help="Resize frames to (width, height)",
    )
    parser.add_argument(
        "--max-ocr-attempts",
        type=int,
        default=10,
        help="Max OCR attempts per vehicle (default: 10)",
    )
    parser.add_argument(
        "--min-plate-area",
        type=int,
        default=500,
        help="Minimum plate area in pixels (default: 500)",
    )

    # Violation settings
    parser.add_argument(
        "--violation-config",
        type=str,
        default="config/violation_rules.json",
        help="Path to violation rules config (default: config/violation_rules.json)",
    )

    # Display settings
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without GUI display (headless mode)",
    )
    parser.add_argument(
        "--save-plates",
        action="store_true",
        help="Save cropped plate images to output/plates/",
    )

    return parser.parse_args()


def main():
    """Entry point for the traffic monitoring system."""
    args = parse_arguments()

    # Ensure output directories exist
    os.makedirs("output/plates", exist_ok=True)
    os.makedirs("output/violations", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)

    # Create and run the system
    system = TrafficMonitoringSystem(args)
    system.run()


if __name__ == "__main__":
    main()
