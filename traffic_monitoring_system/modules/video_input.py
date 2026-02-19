"""
Video Input Module
===================
Handles video input from multiple sources: video files, webcam, and RTSP streams.
Provides a unified interface for frame capture with optional resizing and FPS control.
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple, Generator


class VideoInput:
    """
    Unified video input handler supporting multiple sources.
    
    Supported sources:
        - Video file (MP4, AVI, MKV, etc.)
        - Webcam (device index)
        - RTSP/RTMP stream URL
    """

    def __init__(
        self,
        source,
        target_fps: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize video input.

        Args:
            source: Video file path (str), webcam index (int), or RTSP URL (str)
            target_fps: Target frames per second (None = use source FPS)
            resize: Optional (width, height) to resize frames
        """
        self.source = source
        self.target_fps = target_fps
        self.resize = resize
        self.cap = None
        self.source_fps = 30
        self.frame_count = 0
        self.source_type = self._determine_source_type()

    def _determine_source_type(self) -> str:
        """Determine the type of video source."""
        if isinstance(self.source, int):
            return "webcam"
        elif isinstance(self.source, str):
            if self.source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
                return "stream"
            else:
                return "file"
        return "unknown"

    def open(self) -> bool:
        """
        Open the video source.

        Returns:
            True if successfully opened, False otherwise
        """
        try:
            if self.source_type == "stream":
                # Use FFMPEG backend for streams
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                print(f"[VideoInput] Error: Cannot open source: {self.source}")
                return False

            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                self.source_fps = 30  # Default fallback

            if self.target_fps is None:
                self.target_fps = int(self.source_fps)

            print(f"[VideoInput] Opened {self.source_type}: {self.source}")
            print(f"[VideoInput] Source FPS: {self.source_fps:.1f}, Target FPS: {self.target_fps}")
            print(f"[VideoInput] Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

            return True

        except Exception as e:
            print(f"[VideoInput] Error opening source: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.

        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        self.frame_count += 1

        if self.resize:
            frame = cv2.resize(frame, self.resize)

        return True, frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames with FPS control.

        Yields:
            Video frames as numpy arrays
        """
        if self.cap is None:
            if not self.open():
                return

        frame_interval = 1.0 / max(self.target_fps, 1)

        while True:
            start_time = time.time()

            ret, frame = self.read()
            if not ret:
                break

            yield frame

            # FPS control
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_info(self) -> dict:
        """Get video source information."""
        if self.cap is None:
            return {}

        return {
            "source": str(self.source),
            "source_type": self.source_type,
            "fps": self.source_fps,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "frame_count": self.frame_count,
        }

    def release(self):
        """Release the video source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[VideoInput] Source released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def __del__(self):
        self.release()
