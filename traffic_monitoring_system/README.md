# AI Traffic Violation & Number Plate Detection System

This is a production-level intelligent traffic monitoring system designed and implemented by Manus AI. The system detects vehicles, reads license plates, identifies vehicle colors, tracks vehicles across frames, detects traffic violations, and logs all data to a CSV file with a real-time dashboard display.

## Features

- **Vehicle Detection**: Utilizes YOLOv8 to detect cars, motorcycles, buses, and trucks.
- **License Plate Detection & OCR**: A two-stage process first detects the license plate region and then uses EasyOCR to read the plate number.
- **Vehicle Color Detection**: Identifies the dominant color of the vehicle using HSV color space analysis.
- **Multi-Object Tracking**: Implements the SORT algorithm to track vehicles across frames, assigning a unique ID to each vehicle to prevent duplicate processing.
- **Traffic Violation Detection**: A configurable rules engine detects violations such as wrong-way driving and lane violations.
- **CSV Logging**: All detected information, including violations, is logged to a CSV file with automatic duplicate prevention.
- **Real-Time Dashboard**: An OpenCV-based dashboard displays the video feed with overlays for bounding boxes, track IDs, plate numbers, vehicle colors, and violation alerts.

## Folder Structure

```
/traffic_monitoring_system
├── config/
│   ├── system_config.json
│   └── violation_rules.json
├── data/
│   └── (video files)
├── models/
│   └── (downloaded model weights)
├── modules/
│   ├── __init__.py
│   ├── color_detector.py
│   ├── csv_logger.py
│   ├── dashboard.py
│   ├── ocr_reader.py
│   ├── plate_detector.py
│   ├── tracker.py
│   ├── vehicle_detector.py
│   ├── video_input.py
│   └── violation_detector.py
├── output/
│   ├── logs/
│   ├── plates/
│   └── violations/
├── utils/
│   └── __init__.py
├── main.py
├── requirements.txt
├── test_system.py
└── README.md
```

## How to Run

1.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the System**:

    The `main.py` script is the entry point for the application. You can provide a video source using the `--source` argument.

    -   **From a video file**:

        ```bash
        python main.py --source path/to/your/video.mp4
        ```

    -   **From a webcam**:

        ```bash
        python main.py --source 0
        ```

    -   **From an RTSP stream**:

        ```bash
        python main.py --source rtsp://your_stream_url
        ```

    -   **Headless mode (no display)**:

        To run without the GUI display and save the output to a video file:

        ```bash
        python main.py --source video.mp4 --no-display --output-video output/result.mp4
        ```

## Configuration

-   **System Configuration**: The `config/system_config.json` file contains settings for models, detection thresholds, and I/O paths.
-   **Violation Rules**: The `config/violation_rules.json` file allows you to enable/disable specific violations and define custom rules, such as restricted zones or wrong-way driving directions.

## Modules Overview

-   `main.py`: The main orchestrator that runs the entire pipeline.
-   `modules/vehicle_detector.py`: Handles vehicle detection using YOLOv8.
-   `modules/plate_detector.py`: Detects license plate regions using image processing.
-   `modules/ocr_reader.py`: Reads text from plate images using EasyOCR.
-   `modules/color_detector.py`: Determines the dominant color of a vehicle.
-   `modules/tracker.py`: Implements the SORT algorithm for vehicle tracking.
-   `modules/violation_detector.py`: Detects violations based on the configured rules.
-   `modules/csv_logger.py`: Logs all data to a CSV file.
-   `modules/video_input.py`: Provides a unified interface for video sources.
-   `modules/dashboard.py`: Renders the real-time display.
-   `test_system.py`: A script to validate that all modules are functioning correctly.
