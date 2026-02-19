"""
Traffic Violation Detection Module
====================================
Detects traffic violations based on vehicle behavior, position, and rules.
Supports both automatic detection and manual rule configuration.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class ViolationDetector:
    """
    Detects traffic violations using configurable rules and motion analysis.
    
    Supported automatic violations:
        - Wrong way driving (based on motion direction)
        - Lane violation (based on position boundaries)
        - No helmet detection (for motorcycles, requires additional model)
        - Red light crossing (based on signal zone)
    
    Supports user-defined custom rules via configuration file.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the violation detector.

        Args:
            config_path: Path to JSON configuration file for rules.
                         If None, uses default rules.
        """
        # Default configuration
        self.config = {
            "enabled_violations": [
                "wrong_way",
                "lane_violation",
                "speed_estimation",
                "no_helmet",
            ],
            "wrong_way": {
                "expected_direction": "down",  # 'up', 'down', 'left', 'right'
                "direction_threshold": 30,  # pixels of movement to determine direction
            },
            "lane_violation": {
                "lane_boundaries": [],  # List of (x_start, x_end) lane boundaries
                "restricted_lanes": [],  # Lane indices restricted for certain vehicle types
            },
            "signal_zone": {
                "enabled": False,
                "zone": None,  # (x1, y1, x2, y2) red light zone
                "active": False,  # Whether red light is currently active
            },
            "custom_rules": [],
        }

        # Load config from file if provided
        if config_path:
            self._load_config(config_path)

        # Track history for motion analysis
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.track_violations: Dict[int, List[str]] = defaultdict(list)
        self.max_history = 30  # Maximum position history per track

    def _load_config(self, config_path: str):
        """Load violation rules from a JSON configuration file."""
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            # Merge with defaults
            for key, value in user_config.items():
                if key in self.config:
                    if isinstance(value, dict) and isinstance(self.config[key], dict):
                        self.config[key].update(value)
                    else:
                        self.config[key] = value
                else:
                    self.config[key] = value
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[ViolationDetector] Warning: Could not load config: {e}")

    def update_track(self, track_id: int, center: Tuple[int, int]):
        """
        Update position history for a tracked vehicle.

        Args:
            track_id: Unique track identifier
            center: (cx, cy) center position of vehicle
        """
        history = self.track_history[track_id]
        history.append(center)
        if len(history) > self.max_history:
            history.pop(0)

    def detect_violations(
        self,
        track_id: int,
        vehicle_type: str,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int] = None,
    ) -> List[str]:
        """
        Check for violations for a specific tracked vehicle.

        Args:
            track_id: Unique track identifier
            vehicle_type: Type of vehicle ('Car', 'Motorcycle', 'Bus', 'Truck')
            bbox: (x1, y1, x2, y2) bounding box
            frame_shape: (height, width) of the frame

        Returns:
            List of violation type strings detected
        """
        violations = []
        enabled = self.config.get("enabled_violations", [])
        history = self.track_history.get(track_id, [])

        # Wrong way detection
        if "wrong_way" in enabled and len(history) >= 5:
            if self._check_wrong_way(history):
                violations.append("Wrong Way")

        # Lane violation detection
        if "lane_violation" in enabled:
            lane_config = self.config.get("lane_violation", {})
            if lane_config.get("lane_boundaries"):
                if self._check_lane_violation(bbox, vehicle_type, lane_config):
                    violations.append("Lane Violation")

        # Red light crossing
        signal_config = self.config.get("signal_zone", {})
        if signal_config.get("enabled") and signal_config.get("active"):
            if self._check_red_light(bbox, signal_config):
                violations.append("Red Light Crossing")

        # No helmet (for motorcycles)
        if "no_helmet" in enabled and vehicle_type == "Motorcycle":
            # Note: Full helmet detection requires a dedicated model.
            # This is a placeholder for integration with a helmet detection model.
            pass

        # Custom rules
        for rule in self.config.get("custom_rules", []):
            if self._check_custom_rule(track_id, vehicle_type, bbox, rule):
                violations.append(rule.get("name", "Custom Violation"))

        # Store violations for this track (avoid duplicates per track)
        for v in violations:
            if v not in self.track_violations[track_id]:
                self.track_violations[track_id].append(v)

        return violations

    def _check_wrong_way(self, history: List[Tuple[int, int]]) -> bool:
        """
        Check if vehicle is moving in the wrong direction.

        Args:
            history: List of (cx, cy) positions over time

        Returns:
            True if wrong way detected
        """
        if len(history) < 5:
            return False

        expected = self.config["wrong_way"]["expected_direction"]
        threshold = self.config["wrong_way"]["direction_threshold"]

        # Calculate net movement
        start = history[0]
        end = history[-1]
        dy = end[1] - start[1]
        dx = end[0] - start[0]

        # Determine actual direction
        if abs(dy) > abs(dx):
            actual = "down" if dy > 0 else "up"
        else:
            actual = "right" if dx > 0 else "left"

        # Check if movement is significant enough
        movement = max(abs(dx), abs(dy))
        if movement < threshold:
            return False

        # Compare with expected direction
        return actual != expected

    def _check_lane_violation(
        self,
        bbox: Tuple[int, int, int, int],
        vehicle_type: str,
        lane_config: Dict,
    ) -> bool:
        """
        Check if vehicle is in a restricted lane.

        Args:
            bbox: Vehicle bounding box
            vehicle_type: Type of vehicle
            lane_config: Lane configuration dictionary

        Returns:
            True if lane violation detected
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2

        boundaries = lane_config.get("lane_boundaries", [])
        restricted = lane_config.get("restricted_lanes", [])

        for i, (lane_start, lane_end) in enumerate(boundaries):
            if lane_start <= cx <= lane_end:
                # Check if this lane is restricted for this vehicle type
                for restriction in restricted:
                    if (
                        restriction.get("lane_index") == i
                        and vehicle_type in restriction.get("restricted_types", [])
                    ):
                        return True
        return False

    def _check_red_light(
        self,
        bbox: Tuple[int, int, int, int],
        signal_config: Dict,
    ) -> bool:
        """
        Check if vehicle crossed into the red light zone.

        Args:
            bbox: Vehicle bounding box
            signal_config: Signal zone configuration

        Returns:
            True if red light crossing detected
        """
        zone = signal_config.get("zone")
        if zone is None:
            return False

        zx1, zy1, zx2, zy2 = zone
        x1, y1, x2, y2 = bbox

        # Check overlap with signal zone
        overlap_x = max(0, min(x2, zx2) - max(x1, zx1))
        overlap_y = max(0, min(y2, zy2) - max(y1, zy1))

        return overlap_x > 0 and overlap_y > 0

    def _check_custom_rule(
        self,
        track_id: int,
        vehicle_type: str,
        bbox: Tuple[int, int, int, int],
        rule: Dict,
    ) -> bool:
        """
        Evaluate a custom violation rule.

        Args:
            track_id: Track identifier
            vehicle_type: Vehicle type
            bbox: Vehicle bounding box
            rule: Custom rule dictionary

        Returns:
            True if custom rule violation detected
        """
        rule_type = rule.get("type", "")

        if rule_type == "restricted_zone":
            zone = rule.get("zone", None)
            if zone:
                zx1, zy1, zx2, zy2 = zone
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    # Check if vehicle type is restricted
                    restricted_types = rule.get("vehicle_types", [])
                    if not restricted_types or vehicle_type in restricted_types:
                        return True

        elif rule_type == "wrong_turn":
            # Check if vehicle made an unexpected turn
            history = self.track_history.get(track_id, [])
            if len(history) >= 10:
                # Calculate direction change
                mid = len(history) // 2
                dir1 = np.array(history[mid]) - np.array(history[0])
                dir2 = np.array(history[-1]) - np.array(history[mid])

                if np.linalg.norm(dir1) > 10 and np.linalg.norm(dir2) > 10:
                    cos_angle = np.dot(dir1, dir2) / (
                        np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-6
                    )
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    min_angle = rule.get("min_turn_angle", 60)
                    if angle > min_angle:
                        return True

        return False

    def get_track_violations(self, track_id: int) -> List[str]:
        """Get all recorded violations for a track."""
        return self.track_violations.get(track_id, [])

    def reset(self):
        """Reset all tracking and violation history."""
        self.track_history.clear()
        self.track_violations.clear()
