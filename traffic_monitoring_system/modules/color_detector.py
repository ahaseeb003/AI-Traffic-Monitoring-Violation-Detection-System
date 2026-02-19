"""
Vehicle Color Detection Module
===============================
Extracts the dominant color of a vehicle from its bounding box region.
Uses HSV color space analysis with histogram-based clustering.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from collections import Counter


# HSV color ranges for common vehicle colors
# Format: (lower_bound, upper_bound) in HSV
COLOR_RANGES = {
    "Red": [
        ((0, 70, 50), (10, 255, 255)),
        ((170, 70, 50), (180, 255, 255)),
    ],
    "Orange": [
        ((10, 70, 50), (25, 255, 255)),
    ],
    "Yellow": [
        ((25, 70, 50), (35, 255, 255)),
    ],
    "Green": [
        ((35, 70, 50), (85, 255, 255)),
    ],
    "Blue": [
        ((85, 70, 50), (130, 255, 255)),
    ],
    "Purple": [
        ((130, 70, 50), (170, 255, 255)),
    ],
    "White": [
        ((0, 0, 180), (180, 30, 255)),
    ],
    "Silver": [
        ((0, 0, 120), (180, 30, 180)),
    ],
    "Gray": [
        ((0, 0, 60), (180, 30, 120)),
    ],
    "Black": [
        ((0, 0, 0), (180, 30, 60)),
    ],
}


class ColorDetector:
    """
    Detects the dominant color of a vehicle using HSV color space analysis.
    
    Uses a combination of HSV range matching and pixel counting
    to determine the most prominent color in the vehicle region.
    """

    def __init__(self, sample_ratio: float = 0.6):
        """
        Initialize the color detector.

        Args:
            sample_ratio: Ratio of center region to sample (0.0-1.0).
                          Helps avoid background pixels at edges.
        """
        self.sample_ratio = sample_ratio
        self.color_ranges = COLOR_RANGES

    def detect_color(self, vehicle_crop: np.ndarray) -> str:
        """
        Detect the dominant color of a vehicle.

        Args:
            vehicle_crop: Cropped vehicle image (BGR)

        Returns:
            Color name string (e.g., 'Red', 'Blue', 'White')
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "Unknown"

        # Sample center region to avoid background
        h, w = vehicle_crop.shape[:2]
        margin_y = int(h * (1 - self.sample_ratio) / 2)
        margin_x = int(w * (1 - self.sample_ratio) / 2)
        center_crop = vehicle_crop[
            margin_y : h - margin_y, margin_x : w - margin_x
        ]

        if center_crop.size == 0:
            center_crop = vehicle_crop

        # Convert to HSV
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)

        # Count pixels for each color range
        color_scores = {}
        total_pixels = hsv.shape[0] * hsv.shape[1]

        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                partial_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, partial_mask)

            pixel_count = cv2.countNonZero(mask)
            color_scores[color_name] = pixel_count / max(total_pixels, 1)

        if not color_scores:
            return "Unknown"

        # Return color with highest pixel ratio
        dominant_color = max(color_scores, key=color_scores.get)

        # If no color has significant presence, return Unknown
        if color_scores[dominant_color] < 0.05:
            return "Unknown"

        return dominant_color

    def detect_color_kmeans(
        self, vehicle_crop: np.ndarray, k: int = 3
    ) -> str:
        """
        Alternative color detection using K-means clustering.
        More accurate but slower than HSV range method.

        Args:
            vehicle_crop: Cropped vehicle image (BGR)
            k: Number of color clusters

        Returns:
            Dominant color name string
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "Unknown"

        # Sample center region
        h, w = vehicle_crop.shape[:2]
        margin_y = int(h * (1 - self.sample_ratio) / 2)
        margin_x = int(w * (1 - self.sample_ratio) / 2)
        center_crop = vehicle_crop[
            margin_y : h - margin_y, margin_x : w - margin_x
        ]

        if center_crop.size == 0:
            center_crop = vehicle_crop

        # Resize for speed
        small = cv2.resize(center_crop, (50, 50))
        pixels = small.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.2,
        )
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Find dominant cluster
        label_counts = Counter(labels.flatten())
        dominant_label = label_counts.most_common(1)[0][0]
        dominant_bgr = centers[dominant_label].astype(np.uint8)

        # Convert dominant BGR to HSV and match to color name
        hsv_pixel = cv2.cvtColor(
            np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV
        )[0][0]

        return self._hsv_to_color_name(hsv_pixel)

    def _hsv_to_color_name(self, hsv: np.ndarray) -> str:
        """
        Map an HSV value to the nearest color name.

        Args:
            hsv: HSV pixel value (H, S, V)

        Returns:
            Color name string
        """
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        # Check achromatic colors first
        if v < 60:
            return "Black"
        if s < 30:
            if v > 180:
                return "White"
            elif v > 120:
                return "Silver"
            else:
                return "Gray"

        # Chromatic colors by hue
        if h < 10 or h >= 170:
            return "Red"
        elif h < 25:
            return "Orange"
        elif h < 35:
            return "Yellow"
        elif h < 85:
            return "Green"
        elif h < 130:
            return "Blue"
        else:
            return "Purple"
