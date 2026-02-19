"""
Vehicle Tracking Module
========================
Implements SORT (Simple Online and Realtime Tracking) for multi-object tracking.
Assigns persistent IDs to vehicles across frames to avoid duplicate processing.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from filterpy.kalman import KalmanFilter


def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bb_test: [x1, y1, x2, y2] test bounding box
        bb_gt: [x1, y1, x2, y2] ground truth bounding box

    Returns:
        IoU value between 0 and 1
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])

    union = area_test + area_gt - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def linear_assignment(cost_matrix: np.ndarray):
    """
    Solve the linear assignment problem using scipy.

    Args:
        cost_matrix: Cost matrix for assignment

    Returns:
        Array of matched (row, col) pairs
    """
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(row_ind, col_ind)))
    except ImportError:
        # Fallback: greedy assignment
        matches = []
        used_cols = set()
        for i in range(cost_matrix.shape[0]):
            best_j = -1
            best_cost = float("inf")
            for j in range(cost_matrix.shape[1]):
                if j not in used_cols and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                matches.append([i, best_j])
                used_cols.add(best_j)
        return np.array(matches) if matches else np.empty((0, 2), dtype=int)


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, x2, y2] to [cx, cy, area, aspect_ratio].
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    area = w * h
    ratio = w / max(h, 1e-6)
    return np.array([cx, cy, area, ratio]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """
    Convert [cx, cy, area, aspect_ratio] to [x1, y1, x2, y2].
    """
    w = np.sqrt(max(x[2] * x[3], 0))
    h = max(x[2] / max(w, 1e-6), 0)
    return np.array([
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]).flatten()


class KalmanBoxTracker:
    """
    Represents the internal state of an individual tracked object using a Kalman filter.
    """

    count = 0

    def __init__(self, bbox: np.ndarray):
        """
        Initialize a tracker with a bounding box.

        Args:
            bbox: [x1, y1, x2, y2] initial bounding box
        """
        # State: [cx, cy, area, ratio, vx, vy, va]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Store additional metadata
        self.metadata: Dict = {}

    def update(self, bbox: np.ndarray):
        """
        Update the state with an observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """
        Advance the state and return the predicted bounding box.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """Return the current bounding box estimate."""
        return convert_x_to_bbox(self.kf.x)


class VehicleTracker:
    """
    SORT-based multi-object tracker for vehicles.
    
    Maintains persistent track IDs across frames and provides
    methods to associate detections with existing tracks.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize the vehicle tracker.

        Args:
            max_age: Maximum frames to keep a track without updates
            min_hits: Minimum hits before a track is considered confirmed
            iou_threshold: Minimum IoU for detection-track association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

        # Track which IDs have been "logged" (for duplicate prevention)
        self.logged_ids = set()

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections and return tracked objects.

        Args:
            detections: List of detection dicts with 'bbox' key

        Returns:
            List of tracked object dicts with 'track_id', 'bbox', and
            'is_new' (True if first time confirmed) keys
        """
        self.frame_count += 1

        # Convert detections to numpy array
        if len(detections) > 0:
            dets = np.array([d["bbox"] for d in detections])
        else:
            dets = np.empty((0, 4))

        # Predict new locations of existing trackers
        predicted = []
        to_delete = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_delete.append(i)
            else:
                predicted.append(pos)

        for i in reversed(to_delete):
            self.trackers.pop(i)

        predicted = np.array(predicted) if predicted else np.empty((0, 4))

        # Associate detections with trackers
        matched, unmatched_dets, unmatched_trks = self._associate(
            dets, predicted
        )

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(dets[det_idx])
            # Store detection metadata
            if det_idx < len(detections):
                self.trackers[trk_idx].metadata = {
                    k: v
                    for k, v in detections[det_idx].items()
                    if k != "bbox"
                }

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(dets[det_idx])
            if det_idx < len(detections):
                trk.metadata = {
                    k: v
                    for k, v in detections[det_idx].items()
                    if k != "bbox"
                }
            self.trackers.append(trk)

        # Build output
        tracked_objects = []
        for trk in reversed(self.trackers):
            bbox = trk.get_state().astype(int)

            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
                continue

            # Only return confirmed tracks
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                is_new = trk.id not in self.logged_ids
                if is_new:
                    self.logged_ids.add(trk.id)

                tracked_objects.append({
                    "track_id": trk.id,
                    "bbox": tuple(bbox),
                    "is_new": is_new,
                    "time_since_update": trk.time_since_update,
                    **trk.metadata,
                })

        return tracked_objects

    def _associate(
        self,
        detections: np.ndarray,
        trackers: np.ndarray,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Associate detections with existing trackers using IoU.

        Returns:
            matched: Array of (det_idx, trk_idx) pairs
            unmatched_dets: List of unmatched detection indices
            unmatched_trks: List of unmatched tracker indices
        """
        if len(trackers) == 0:
            return (
                np.empty((0, 2), dtype=int),
                list(range(len(detections))),
                [],
            )

        if len(detections) == 0:
            return (
                np.empty((0, 2), dtype=int),
                [],
                list(range(len(trackers))),
            )

        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d in range(len(detections)):
            for t in range(len(trackers)):
                iou_matrix[d, t] = iou(detections[d], trackers[t])

        # Hungarian algorithm (minimize cost = 1 - IoU)
        cost_matrix = 1.0 - iou_matrix
        if cost_matrix.size > 0:
            matched_indices = linear_assignment(cost_matrix)
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))
        matched = []

        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= self.iou_threshold:
                matched.append(m)
                if m[0] in unmatched_dets:
                    unmatched_dets.remove(m[0])
                if m[1] in unmatched_trks:
                    unmatched_trks.remove(m[1])

        return (
            np.array(matched) if matched else np.empty((0, 2), dtype=int),
            unmatched_dets,
            unmatched_trks,
        )

    def reset(self):
        """Reset the tracker state."""
        self.trackers = []
        self.frame_count = 0
        self.logged_ids = set()
        KalmanBoxTracker.count = 0
