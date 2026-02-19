# AI Traffic Monitoring & Violation Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

This project is an advanced, production-level intelligent traffic monitoring system. What started as a 2nd-semester Object-Oriented Programming (OOP) project has been completely re-engineered and enhanced with a powerful AI pipeline to create a comprehensive solution for real-time traffic analysis.

This system detects and tracks vehicles, reads license plates, identifies vehicle colors, and automatically flags traffic violations from any video source, including local files, webcams, and live YouTube streams.

## Project Journey

This project began as a conceptual model in a 2nd-semester OOP course, with the initial idea and structure developed by **Musa Khan**. It has since evolved into a full-fledged AI application, demonstrating the power of applying modern computer vision and deep learning techniques to a foundational software design. Special thanks to **Ayesha Khalid** and others for their continuous support and encouragement throughout this journey.

## System Architecture

The system is designed with a modular and scalable architecture. The data flows through a pipeline of specialized modules, each responsible for a specific task, from video input to final data logging and display.

#### Main Workflow

*(To display this image, create a `diagrams` folder in your repository and upload the `system_workflow.png` file I provided earlier.)*

`![System Workflow](diagrams/system_workflow.png)`

## Features

| Feature | Description | Technology |
| :--- | :--- | :--- |
| **Vehicle Detection** | Detects multiple classes of vehicles (cars, motorbikes, buses, trucks). | YOLOv8 |
| **License Plate Recognition** | A two-stage process detects the plate region and then reads the text. | Contour Detection, EasyOCR |
| **Vehicle Color Detection** | Identifies the dominant color of each vehicle for easier identification. | HSV Color Space Analysis |
| **Multi-Object Tracking** | Assigns and maintains a unique ID for each vehicle across frames. | SORT Algorithm |
| **Violation Detection** | A configurable rules engine detects violations like wrong-way driving. | Custom Logic |
| **Data Logging** | Logs all detected vehicle data and violations to a CSV file. | Python CSV Module |
| **Real-Time Dashboard** | Renders a live display with rich visual overlays for all detected information. | OpenCV |
| **Flexible Video Input** | Processes video from local files, webcams, RTSP streams, and YouTube URLs. | `pafy`, `yt-dlp` |

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` and `venv`
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ahaseeb003/AI-Traffic-Monitoring-Violation-Detection-System.git
    cd AI-Traffic-Monitoring-Violation-Detection-System
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The system requires several computer vision and deep learning libraries. For YouTube URL support, `yt-dlp` and `pafy` are also required.
    ```bash
    pip install -r requirements.txt
    pip install yt-dlp pafy
    ```

4.  **Windows SSL Certificate Fix (Important):**
    On the first run, `easyocr` needs to download models, which can fail on Windows due to SSL errors. The code includes a fix, but if you still encounter issues, please run the following commands:
    ```bash
    pip install --upgrade certifi
    pip install --upgrade pip setuptools
    ```

### How to Run

The `main.py` script is the main entry point. The `--source` argument is highly flexible.

-   **From a local video file:**
    ```bash
    python main.py --source data/traffic_test_video.mp4
    ```

-   **From a YouTube video:**
    ```bash
    python main.py --source "https://www.youtube.com/watch?v=wH7S6VPQ4V4"
    ```

-   **From a webcam:**
    ```bash
    python main.py --source 0
    ```

-   **Headless mode (no display):**
    To run without a GUI and save the processed video to a file:
    ```bash
    python main.py --source data/traffic_test_video.mp4 --no-display --output-video output/result.mp4
    ```

### Command-Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--source` | `0` | Video source (file, webcam ID, RTSP/YouTube URL). |
| `--save-plates` | `False` | Save cropped images of detected license plates. |
| `--no-display` | `False` | Run in headless mode without a GUI window. |
| `--output-video` | `None` | Path to save the output video with overlays. |
| `--confidence` | `0.4` | YOLOv8 detection confidence threshold. |
| `--device` | `cpu` | Inference device (`cpu`, `cuda`, `mps`). |

## Configuration

System behavior can be customized via JSON configuration files in the `config/` directory.

-   `system_config.json`: Main settings for models, devices, and thresholds.
-   `violation_rules.json`: Define and enable/disable specific traffic violations. You can set restricted zones, expected traffic flow direction, and more.

## Project Structure

```
/AI-Traffic-Monitoring-Violation-Detection-System
├── config/
│   ├── system_config.json
│   └── violation_rules.json
├── data/
│   └── (video files)
├── diagrams/
│   └── (workflow diagrams)
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
├── main.py
├── requirements.txt
├── test_system.py
├── LICENSE
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

-   **Musa Khan**: For the original OOP project idea and structure.
-   **Ayesha Khalid**: For continuous support and encouragement.
-   The teams behind **YOLOv8**, **EasyOCR**, and **SORT** for their incredible open-source tools.
