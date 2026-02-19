"""
Real-Time Dashboard Module
============================
Provides visual overlay on video frames with detection results,
bounding boxes, plate numbers, violations, and vehicle counts.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime


# Color palette for visualization (BGR format)
COLORS = {
    "Car": (0, 255, 0),        # Green
    "Motorcycle": (255, 165, 0), # Orange
    "Bus": (255, 0, 0),         # Blue
    "Truck": (0, 0, 255),       # Red
    "violation": (0, 0, 255),   # Red
    "plate": (255, 255, 0),     # Cyan
    "text_bg": (0, 0, 0),       # Black
    "text": (255, 255, 255),    # White
    "header_bg": (50, 50, 50),  # Dark gray
}


class Dashboard:
    """
    Real-time visual dashboard for the traffic monitoring system.
    
    Renders:
        - Vehicle bounding boxes with class labels
        - License plate numbers
        - Violation labels
        - Running vehicle counts
        - System status information
    """

    def __init__(
        self,
        show_counts: bool = True,
        show_fps: bool = True,
        show_timestamp: bool = True,
        font_scale: float = 0.6,
        thickness: int = 2,
    ):
        """
        Initialize the dashboard.

        Args:
            show_counts: Whether to display vehicle counts panel
            show_fps: Whether to display FPS counter
            show_timestamp: Whether to display current timestamp
            font_scale: Font scale for text rendering
            thickness: Line thickness for bounding boxes
        """
        self.show_counts = show_counts
        self.show_fps = show_fps
        self.show_timestamp = show_timestamp
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # FPS calculation
        self.fps = 0.0
        self.frame_times = []

    def render(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict],
        vehicle_counts: Dict[str, int],
        total_logged: int = 0,
    ) -> np.ndarray:
        """
        Render the dashboard overlay on a frame.

        Args:
            frame: Input BGR frame
            tracked_objects: List of tracked object dictionaries
            vehicle_counts: Dictionary of vehicle type counts
            total_logged: Total number of logged entries

        Returns:
            Frame with dashboard overlay
        """
        display = frame.copy()

        # Draw tracked objects
        for obj in tracked_objects:
            self._draw_vehicle(display, obj)

        # Draw counts panel
        if self.show_counts:
            self._draw_counts_panel(display, vehicle_counts, total_logged)

        # Draw FPS
        if self.show_fps:
            self._draw_fps(display)

        # Draw timestamp
        if self.show_timestamp:
            self._draw_timestamp(display)

        # Draw title bar
        self._draw_title(display)

        return display

    def _draw_vehicle(self, frame: np.ndarray, obj: Dict):
        """Draw bounding box and labels for a tracked vehicle."""
        bbox = obj.get("bbox", (0, 0, 0, 0))
        x1, y1, x2, y2 = [int(v) for v in bbox]
        track_id = obj.get("track_id", -1)
        class_name = obj.get("class_name", "Unknown")
        plate_number = obj.get("plate_number", "")
        violations = obj.get("violations", [])
        color_name = obj.get("vehicle_color", "")
        confidence = obj.get("confidence", 0.0)

        # Choose color based on vehicle type
        box_color = COLORS.get(class_name, (0, 255, 0))

        # If violation, use red
        has_violation = violations and any(v != "None" for v in violations)
        if has_violation:
            box_color = COLORS["violation"]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, self.thickness)

        # Build label text
        label_parts = [f"ID:{track_id}", class_name]
        if color_name:
            label_parts.append(color_name)

        label = " | ".join(label_parts)

        # Draw label background
        label_size = cv2.getTextSize(label, self.font, self.font_scale * 0.8, 1)[0]
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            box_color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            self.font,
            self.font_scale * 0.8,
            COLORS["text"],
            1,
            cv2.LINE_AA,
        )

        # Draw plate number if available
        if plate_number:
            plate_label = f"Plate: {plate_number}"
            plate_size = cv2.getTextSize(
                plate_label, self.font, self.font_scale * 0.7, 1
            )[0]
            cv2.rectangle(
                frame,
                (x1, y2),
                (x1 + plate_size[0] + 10, y2 + plate_size[1] + 10),
                COLORS["plate"],
                -1,
            )
            cv2.putText(
                frame,
                plate_label,
                (x1 + 5, y2 + plate_size[1] + 5),
                self.font,
                self.font_scale * 0.7,
                COLORS["text_bg"],
                1,
                cv2.LINE_AA,
            )

        # Draw violation labels
        if has_violation:
            violation_text = "VIOLATION: " + ", ".join(violations)
            viol_size = cv2.getTextSize(
                violation_text, self.font, self.font_scale * 0.7, 1
            )[0]
            vy = y2 + 30 if plate_number else y2 + 5
            cv2.rectangle(
                frame,
                (x1, vy),
                (x1 + viol_size[0] + 10, vy + viol_size[1] + 10),
                COLORS["violation"],
                -1,
            )
            cv2.putText(
                frame,
                violation_text,
                (x1 + 5, vy + viol_size[1] + 5),
                self.font,
                self.font_scale * 0.7,
                COLORS["text"],
                1,
                cv2.LINE_AA,
            )

    def _draw_counts_panel(
        self,
        frame: np.ndarray,
        counts: Dict[str, int],
        total_logged: int,
    ):
        """Draw vehicle counts panel in top-right corner."""
        h, w = frame.shape[:2]
        panel_w = 220
        panel_h = 30 + len(counts) * 25 + 30
        px = w - panel_w - 10
        py = 50

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px, py),
            (px + panel_w, py + panel_h),
            COLORS["header_bg"],
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Panel title
        cv2.putText(
            frame,
            "Vehicle Counts",
            (px + 10, py + 20),
            self.font,
            self.font_scale * 0.8,
            COLORS["text"],
            1,
            cv2.LINE_AA,
        )

        # Vehicle counts
        y_offset = py + 45
        total = 0
        for vtype, count in counts.items():
            color = COLORS.get(vtype, (255, 255, 255))
            text = f"{vtype}: {count}"
            cv2.putText(
                frame,
                text,
                (px + 15, y_offset),
                self.font,
                self.font_scale * 0.7,
                color,
                1,
                cv2.LINE_AA,
            )
            y_offset += 25
            total += count

        # Total and logged
        cv2.putText(
            frame,
            f"Total: {total} | Logged: {total_logged}",
            (px + 15, y_offset),
            self.font,
            self.font_scale * 0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_fps(self, frame: np.ndarray):
        """Draw FPS counter."""
        import time

        self.frame_times.append(time.time())
        # Keep last 30 frame times
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        if len(self.frame_times) >= 2:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            self.fps = (len(self.frame_times) - 1) / max(elapsed, 1e-6)

        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (10, frame.shape[0] - 15),
            self.font,
            self.font_scale * 0.8,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    def _draw_timestamp(self, frame: np.ndarray):
        """Draw current timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            timestamp,
            (w - 250, h - 15),
            self.font,
            self.font_scale * 0.7,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_title(self, frame: np.ndarray):
        """Draw title bar at top of frame."""
        h, w = frame.shape[:2]
        # Title bar background
        cv2.rectangle(frame, (0, 0), (w, 40), COLORS["header_bg"], -1)
        cv2.putText(
            frame,
            "AI Traffic Monitoring System",
            (10, 28),
            self.font,
            self.font_scale * 1.0,
            COLORS["text"],
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def show_frame(window_name: str, frame: np.ndarray) -> int:
        """
        Display frame in a window and return key press.

        Args:
            window_name: Window title
            frame: Frame to display

        Returns:
            Key code pressed (or -1 if no key)
        """
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF

    @staticmethod
    def save_frame(path: str, frame: np.ndarray):
        """Save a frame to disk."""
        cv2.imwrite(path, frame)

    @staticmethod
    def destroy_all():
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()
