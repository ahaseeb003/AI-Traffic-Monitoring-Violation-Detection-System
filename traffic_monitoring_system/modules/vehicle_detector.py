"""
Vehicle Detection Module
========================
Uses YOLOv8 for detecting vehicles (car, motorcycle, bus, truck) in video frames.
Provides bounding boxes, class labels, and confidence scores.
"""

import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional


# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# All COCO vehicle class IDs
VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())


class VehicleDetector:
    """
    Detects vehicles in video frames using YOLOv8.
    
    Attributes:
        model: YOLOv8 model instance
        confidence_threshold: Minimum confidence for detection
        vehicle_counts: Running count of detected vehicles by type
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
    ):
        """
        Initialize the vehicle detector.

        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence threshold for detections
            device: Device to run inference on ('cpu', 'cuda', '0', etc.)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.vehicle_counts: Dict[str, int] = {v: 0 for v in VEHICLE_CLASSES.values()}

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detection dictionaries with keys:
                - bbox: (x1, y1, x2, y2) bounding box coordinates
                - class_id: COCO class ID
                - class_name: Human-readable vehicle type
                - confidence: Detection confidence score
                - center: (cx, cy) center point of bounding box
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            classes=VEHICLE_CLASS_IDS,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                class_name = VEHICLE_CLASSES.get(class_id, "Unknown")
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "center": (cx, cy),
                })

        return detections

    def update_counts(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Update running vehicle counts based on new detections.
        Should be called once per tracked (unique) vehicle, not per frame.

        Args:
            detections: List of detection dictionaries

        Returns:
            Updated vehicle counts dictionary
        """
        for det in detections:
            vtype = det.get("class_name", "Unknown")
            if vtype in self.vehicle_counts:
                self.vehicle_counts[vtype] += 1
        return self.vehicle_counts.copy()

    def get_counts(self) -> Dict[str, int]:
        """Return current vehicle counts."""
        return self.vehicle_counts.copy()

    def reset_counts(self):
        """Reset all vehicle counts to zero."""
        self.vehicle_counts = {v: 0 for v in VEHICLE_CLASSES.values()}

    @staticmethod
    def crop_vehicle(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop vehicle region from frame.

        Args:
            frame: Full frame image
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            Cropped vehicle image
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return frame[y1:y2, x1:x2].copy()
