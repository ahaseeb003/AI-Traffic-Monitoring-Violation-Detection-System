"""
CSV Logger Module
==================
Handles logging of vehicle detection and violation data to CSV files.
Supports automatic file creation, duplicate prevention, and proper formatting.
"""

import csv
import os
from datetime import datetime
from typing import Dict, Optional, Set
from threading import Lock


class CSVLogger:
    """
    Logs vehicle detection and violation data to CSV files.
    
    Features:
        - Auto-creates CSV file with headers if not exists
        - Prevents duplicate entries based on plate number + track ID
        - Thread-safe write operations
        - Proper timestamp formatting
        - Auto-incrementing serial numbers
    """

    # CSV column headers
    HEADERS = [
        "S.No",
        "Plate Number",
        "Vehicle Type",
        "Vehicle Color",
        "Violation",
        "Timestamp",
        "Confidence",
        "Track ID",
    ]

    def __init__(
        self,
        output_path: str = "output/detection_log.csv",
        auto_create: bool = True,
    ):
        """
        Initialize the CSV logger.

        Args:
            output_path: Path to the CSV output file
            auto_create: Whether to auto-create the file with headers
        """
        self.output_path = output_path
        self.lock = Lock()
        self.serial_number = 0
        self.logged_entries: Set[str] = set()  # For duplicate prevention

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if auto_create:
            self._initialize_file()

    def _initialize_file(self):
        """Create CSV file with headers if it doesn't exist, or load existing data."""
        if os.path.exists(self.output_path):
            # Load existing entries for duplicate prevention
            try:
                with open(self.output_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.serial_number += 1
                        # Create unique key for duplicate check
                        key = self._make_key(
                            row.get("Plate Number", ""),
                            row.get("Track ID", ""),
                        )
                        self.logged_entries.add(key)
            except Exception:
                pass
        else:
            # Create new file with headers
            with open(self.output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def _make_key(self, plate_number: str, track_id: str) -> str:
        """Create a unique key for duplicate detection."""
        return f"{plate_number}_{track_id}"

    def log_detection(
        self,
        plate_number: str,
        vehicle_type: str,
        vehicle_color: str,
        violation: str = "None",
        confidence: float = 0.0,
        track_id: int = -1,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Log a vehicle detection entry to the CSV file.

        Args:
            plate_number: Detected license plate number
            vehicle_type: Type of vehicle (Car, Motorcycle, Bus, Truck)
            vehicle_color: Detected vehicle color
            violation: Violation type or 'None'
            confidence: OCR confidence score (0.0 - 1.0)
            track_id: Unique track identifier
            timestamp: Optional timestamp string; auto-generated if None

        Returns:
            True if entry was logged, False if duplicate
        """
        # Generate unique key
        key = self._make_key(plate_number, str(track_id))

        with self.lock:
            # Check for duplicates
            if key in self.logged_entries:
                return False

            self.serial_number += 1
            self.logged_entries.add(key)

            if timestamp is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            row = [
                self.serial_number,
                plate_number,
                vehicle_type,
                vehicle_color,
                violation,
                timestamp,
                f"{confidence:.3f}",
                track_id,
            ]

            try:
                with open(self.output_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                return True
            except Exception as e:
                print(f"[CSVLogger] Error writing to CSV: {e}")
                self.serial_number -= 1
                self.logged_entries.discard(key)
                return False

    def log_violation(
        self,
        plate_number: str,
        vehicle_type: str,
        vehicle_color: str,
        violations: list,
        confidence: float = 0.0,
        track_id: int = -1,
    ) -> int:
        """
        Log one or more violations for a vehicle.

        Args:
            plate_number: Detected license plate number
            vehicle_type: Type of vehicle
            vehicle_color: Detected vehicle color
            violations: List of violation type strings
            confidence: OCR confidence score
            track_id: Unique track identifier

        Returns:
            Number of new entries logged
        """
        count = 0
        if not violations:
            violations = ["None"]

        violation_str = ", ".join(violations) if violations else "None"
        if self.log_detection(
            plate_number=plate_number,
            vehicle_type=vehicle_type,
            vehicle_color=vehicle_color,
            violation=violation_str,
            confidence=confidence,
            track_id=track_id,
        ):
            count += 1

        return count

    def get_entry_count(self) -> int:
        """Return the total number of logged entries."""
        return self.serial_number

    def get_logged_plates(self) -> Set[str]:
        """Return set of all logged plate-track keys."""
        return self.logged_entries.copy()

    def reset(self):
        """Reset the logger and create a fresh CSV file."""
        self.serial_number = 0
        self.logged_entries.clear()
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
