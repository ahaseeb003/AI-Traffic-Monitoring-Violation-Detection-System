"""
License Plate Detection Module
===============================
Detects license plate regions within vehicle bounding boxes.
Uses a secondary YOLO model or contour-based detection for plate localization.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class PlateDetector:
    """
    Detects license plate regions in vehicle crops using image processing techniques.
    Falls back to contour-based detection for robustness.
    
    For production use, a dedicated YOLO plate detection model can be loaded.
    """

    def __init__(
        self,
        min_plate_area: int = 500,
        max_plate_area: int = 50000,
        aspect_ratio_range: Tuple[float, float] = (1.5, 6.0),
    ):
        """
        Initialize the plate detector.

        Args:
            min_plate_area: Minimum area threshold for plate candidates
            max_plate_area: Maximum area threshold for plate candidates
            aspect_ratio_range: (min, max) width/height ratio for plates
        """
        self.min_plate_area = min_plate_area
        self.max_plate_area = max_plate_area
        self.aspect_ratio_range = aspect_ratio_range

    def detect_plate(
        self, vehicle_crop: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect license plate in a vehicle crop image.

        Args:
            vehicle_crop: Cropped vehicle image (BGR)

        Returns:
            Tuple of (plate_crop, (x, y, w, h)) relative to vehicle crop,
            or None if no plate found.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edges = cv2.Canny(filtered, 30, 200)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        plate_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_plate_area or area > self.max_plate_area:
                continue

            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # License plates are roughly rectangular (4 corners)
            if len(approx) >= 4 and len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(approx)

                if h == 0:
                    continue

                aspect_ratio = w / h

                if (
                    self.aspect_ratio_range[0]
                    <= aspect_ratio
                    <= self.aspect_ratio_range[1]
                ):
                    plate_candidates.append({
                        "bbox": (x, y, w, h),
                        "area": area,
                        "aspect_ratio": aspect_ratio,
                    })

        # If contour method fails, try morphological approach
        if not plate_candidates:
            plate_candidates = self._morphological_detection(gray)

        if not plate_candidates:
            return None

        # Select best candidate (largest area with good aspect ratio)
        best = max(plate_candidates, key=lambda c: c["area"])
        x, y, w, h = best["bbox"]

        # Add small padding
        pad = 5
        vh, vw = vehicle_crop.shape[:2]
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(vw - x, w + 2 * pad)
        h = min(vh - y, h + 2 * pad)

        plate_crop = vehicle_crop[y : y + h, x : x + w].copy()

        if plate_crop.size == 0:
            return None

        return plate_crop, (x, y, w, h)

    def _morphological_detection(self, gray: np.ndarray) -> List[Dict]:
        """
        Fallback plate detection using morphological operations.

        Args:
            gray: Grayscale image

        Returns:
            List of plate candidate dictionaries
        """
        candidates = []

        # Apply blackhat morphological operation to reveal dark regions on light bg
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # Threshold
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Close gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        # Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_plate_area or area > self.max_plate_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue

            aspect_ratio = w / h
            if (
                self.aspect_ratio_range[0]
                <= aspect_ratio
                <= self.aspect_ratio_range[1]
            ):
                candidates.append({
                    "bbox": (x, y, w, h),
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                })

        return candidates

    @staticmethod
    def preprocess_plate(plate_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for OCR.

        Args:
            plate_crop: Cropped plate image (BGR)

        Returns:
            Preprocessed grayscale plate image
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        # Resize for better OCR
        h, w = plate_crop.shape[:2]
        if w < 100:
            scale = 200 / max(w, 1)
            plate_crop = cv2.resize(
                plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive threshold for better character separation
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Slight dilation to thicken characters
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)

        return processed
