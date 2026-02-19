"""
OCR Reader Module
=================
Performs Optical Character Recognition on license plate crops.
Uses EasyOCR as primary engine with text cleaning and validation.
"""

import re
import os
import ssl
import numpy as np
from typing import Optional, Tuple

# Fix SSL certificate verification errors on Windows
# This is needed when EasyOCR downloads its detection models
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Also set environment variable to disable SSL verification for pip/urllib
os.environ['CURL_CA_BUNDLE'] = ''

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class OCRReader:
    """
    Reads text from license plate images using EasyOCR.
    
    Includes text cleaning, validation, and confidence scoring.
    Handles noisy, blurred, and angled plate images.
    """

    def __init__(
        self,
        languages: list = None,
        gpu: bool = False,
        min_confidence: float = 0.3,
        min_plate_length: int = 3,
        max_plate_length: int = 15,
    ):
        """
        Initialize the OCR reader.

        Args:
            languages: List of language codes for EasyOCR (default: ['en'])
            gpu: Whether to use GPU acceleration
            min_confidence: Minimum confidence threshold for OCR results
            min_plate_length: Minimum valid plate text length
            max_plate_length: Maximum valid plate text length
        """
        if languages is None:
            languages = ["en"]

        self.min_confidence = min_confidence
        self.min_plate_length = min_plate_length
        self.max_plate_length = max_plate_length

        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
            except Exception as e:
                error_msg = str(e)
                if "SSL" in error_msg or "CERTIFICATE" in error_msg or "certificate" in error_msg:
                    print("[OCR] SSL error detected. Attempting to fix...")
                    print("[OCR] TIP: If this persists, run these commands first:")
                    print("      pip install certifi")
                    print("      pip install --upgrade pip setuptools")
                    # Try again with patched SSL
                    import urllib.request
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    opener = urllib.request.build_opener(
                        urllib.request.HTTPSHandler(context=ctx)
                    )
                    urllib.request.install_opener(opener)
                    self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
                else:
                    raise
        else:
            raise ImportError(
                "EasyOCR is not installed. Install with: pip install easyocr"
            )

    def read_plate(
        self, plate_image: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Read text from a license plate image.

        Args:
            plate_image: Preprocessed plate image (grayscale or BGR)

        Returns:
            Tuple of (plate_text, confidence_score).
            Returns (None, 0.0) if no valid text detected.
        """
        if plate_image is None or plate_image.size == 0:
            return None, 0.0

        try:
            results = self.reader.readtext(
                plate_image,
                detail=1,
                paragraph=False,
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz- ",
            )
        except Exception:
            return None, 0.0

        if not results:
            return None, 0.0

        # Combine all detected text segments
        combined_text = ""
        total_confidence = 0.0
        count = 0

        for bbox, text, confidence in results:
            if confidence >= self.min_confidence:
                combined_text += text + " "
                total_confidence += confidence
                count += 1

        if count == 0:
            return None, 0.0

        avg_confidence = total_confidence / count
        cleaned_text = self._clean_plate_text(combined_text.strip())

        # Validate plate text
        if not self._validate_plate(cleaned_text):
            return None, 0.0

        return cleaned_text, round(avg_confidence, 3)

    def _clean_plate_text(self, text: str) -> str:
        """
        Clean and normalize OCR output for license plates.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned plate text
        """
        # Convert to uppercase
        text = text.upper().strip()

        # Common OCR misreads correction
        replacements = {
            "O": "0",  # Only in numeric context
            "I": "1",  # Only in numeric context
            "S": "5",  # Only in numeric context
            "B": "8",  # Only in numeric context
            "Z": "2",  # Only in numeric context
            "G": "6",  # Only in numeric context
        }

        # Remove unwanted characters (keep alphanumeric, dash, space)
        text = re.sub(r"[^A-Z0-9\- ]", "", text)

        # Remove excessive spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Remove leading/trailing dashes
        text = text.strip("-").strip()

        return text

    def _validate_plate(self, text: str) -> bool:
        """
        Validate if text looks like a license plate number.

        Args:
            text: Cleaned plate text

        Returns:
            True if text appears to be a valid plate number
        """
        if not text:
            return False

        # Check length
        text_no_spaces = text.replace(" ", "").replace("-", "")
        if len(text_no_spaces) < self.min_plate_length:
            return False
        if len(text_no_spaces) > self.max_plate_length:
            return False

        # Must contain at least one digit and one letter (most plates)
        has_digit = any(c.isdigit() for c in text_no_spaces)
        has_alpha = any(c.isalpha() for c in text_no_spaces)

        # Accept plates with only digits or only letters too (some formats)
        if not (has_digit or has_alpha):
            return False

        return True

    @staticmethod
    def get_best_result(
        results: list,
    ) -> Tuple[Optional[str], float]:
        """
        From multiple OCR attempts, select the best result.

        Args:
            results: List of (text, confidence) tuples

        Returns:
            Best (text, confidence) tuple
        """
        valid_results = [
            (text, conf) for text, conf in results if text is not None
        ]

        if not valid_results:
            return None, 0.0

        # Return highest confidence result
        return max(valid_results, key=lambda x: x[1])
