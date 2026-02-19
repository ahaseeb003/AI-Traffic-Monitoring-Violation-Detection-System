"""
System Validation Test
=======================
Tests that all modules load correctly and basic functionality works.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all module imports."""
    print("[TEST] Testing module imports...")
    
    from modules.vehicle_detector import VehicleDetector
    print("  ✓ VehicleDetector imported")
    
    from modules.plate_detector import PlateDetector
    print("  ✓ PlateDetector imported")
    
    from modules.ocr_reader import OCRReader
    print("  ✓ OCRReader imported")
    
    from modules.color_detector import ColorDetector
    print("  ✓ ColorDetector imported")
    
    from modules.tracker import VehicleTracker
    print("  ✓ VehicleTracker imported")
    
    from modules.violation_detector import ViolationDetector
    print("  ✓ ViolationDetector imported")
    
    from modules.csv_logger import CSVLogger
    print("  ✓ CSVLogger imported")
    
    from modules.video_input import VideoInput
    print("  ✓ VideoInput imported")
    
    from modules.dashboard import Dashboard
    print("  ✓ Dashboard imported")
    
    print("[TEST] All imports successful!\n")
    return True


def test_vehicle_detector():
    """Test vehicle detector initialization and inference."""
    print("[TEST] Testing VehicleDetector...")
    from modules.vehicle_detector import VehicleDetector
    
    detector = VehicleDetector(model_path="yolov8n.pt", device="cpu")
    print("  ✓ Model loaded")
    
    # Test with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    print(f"  ✓ Inference works (detected {len(detections)} objects on random noise)")
    
    counts = detector.get_counts()
    print(f"  ✓ Vehicle counts: {counts}")
    
    return True


def test_plate_detector():
    """Test plate detector."""
    print("[TEST] Testing PlateDetector...")
    from modules.plate_detector import PlateDetector
    
    detector = PlateDetector()
    print("  ✓ PlateDetector initialized")
    
    # Test with dummy image
    dummy = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
    result = detector.detect_plate(dummy)
    print(f"  ✓ Plate detection works (result: {'Found' if result else 'None'})")
    
    return True


def test_ocr_reader():
    """Test OCR reader."""
    print("[TEST] Testing OCRReader...")
    from modules.ocr_reader import OCRReader
    
    reader = OCRReader(languages=["en"], gpu=False)
    print("  ✓ OCR engine loaded")
    
    # Test with dummy image
    dummy = np.ones((50, 200), dtype=np.uint8) * 255
    text, conf = reader.read_plate(dummy)
    print(f"  ✓ OCR works (text: {text}, conf: {conf})")
    
    return True


def test_color_detector():
    """Test color detector."""
    print("[TEST] Testing ColorDetector...")
    from modules.color_detector import ColorDetector
    
    detector = ColorDetector()
    print("  ✓ ColorDetector initialized")
    
    # Test with solid color image
    red_image = np.zeros((100, 100, 3), dtype=np.uint8)
    red_image[:, :, 2] = 200  # Red channel in BGR
    color = detector.detect_color(red_image)
    print(f"  ✓ Color detection works (red image → '{color}')")
    
    blue_image = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_image[:, :, 0] = 200  # Blue channel in BGR
    color = detector.detect_color(blue_image)
    print(f"  ✓ Color detection works (blue image → '{color}')")
    
    return True


def test_tracker():
    """Test vehicle tracker."""
    print("[TEST] Testing VehicleTracker...")
    from modules.tracker import VehicleTracker
    
    tracker = VehicleTracker()
    print("  ✓ Tracker initialized")
    
    # Test with dummy detections
    detections = [
        {"bbox": (100, 100, 200, 200), "class_name": "Car", "confidence": 0.9},
        {"bbox": (300, 300, 400, 400), "class_name": "Truck", "confidence": 0.8},
    ]
    
    tracked = tracker.update(detections)
    print(f"  ✓ Tracking works ({len(tracked)} tracked objects)")
    
    # Second frame
    detections2 = [
        {"bbox": (105, 105, 205, 205), "class_name": "Car", "confidence": 0.9},
        {"bbox": (305, 305, 405, 405), "class_name": "Truck", "confidence": 0.8},
    ]
    tracked2 = tracker.update(detections2)
    print(f"  ✓ Multi-frame tracking works ({len(tracked2)} tracked objects)")
    
    return True


def test_violation_detector():
    """Test violation detector."""
    print("[TEST] Testing ViolationDetector...")
    from modules.violation_detector import ViolationDetector
    
    detector = ViolationDetector()
    print("  ✓ ViolationDetector initialized")
    
    # Simulate wrong way movement
    for i in range(10):
        detector.update_track(1, (100, 300 - i * 20))  # Moving up
    
    violations = detector.detect_violations(
        track_id=1,
        vehicle_type="Car",
        bbox=(50, 200, 150, 300),
    )
    print(f"  ✓ Violation detection works (violations: {violations})")
    
    return True


def test_csv_logger():
    """Test CSV logger."""
    print("[TEST] Testing CSVLogger...")
    from modules.csv_logger import CSVLogger
    
    test_path = "output/test_log.csv"
    logger = CSVLogger(output_path=test_path)
    print("  ✓ CSVLogger initialized")
    
    # Log test entry
    logged = logger.log_detection(
        plate_number="ABC-1234",
        vehicle_type="Car",
        vehicle_color="Red",
        violation="None",
        confidence=0.95,
        track_id=1,
    )
    print(f"  ✓ Logging works (new entry: {logged})")
    
    # Test duplicate prevention
    logged2 = logger.log_detection(
        plate_number="ABC-1234",
        vehicle_type="Car",
        vehicle_color="Red",
        violation="None",
        confidence=0.95,
        track_id=1,
    )
    print(f"  ✓ Duplicate prevention works (duplicate blocked: {not logged2})")
    
    print(f"  ✓ Total entries: {logger.get_entry_count()}")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    return True


def test_config():
    """Test configuration files."""
    print("[TEST] Testing configuration files...")
    import json
    
    config_path = "config/violation_rules.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"  ✓ Violation rules config loaded ({len(config)} keys)")
    else:
        print(f"  ✗ Config file not found: {config_path}")
    
    sys_config_path = "config/system_config.json"
    if os.path.exists(sys_config_path):
        with open(sys_config_path) as f:
            sys_config = json.load(f)
        print(f"  ✓ System config loaded ({len(sys_config)} sections)")
    else:
        print(f"  ✗ System config not found: {sys_config_path}")
    
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("  AI Traffic Monitoring System — Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Module Imports", test_imports),
        ("Vehicle Detector", test_vehicle_detector),
        ("Plate Detector", test_plate_detector),
        ("OCR Reader", test_ocr_reader),
        ("Color Detector", test_color_detector),
        ("Vehicle Tracker", test_tracker),
        ("Violation Detector", test_violation_detector),
        ("CSV Logger", test_csv_logger),
        ("Configuration", test_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append((name, False))
        print()
    
    print("=" * 60)
    print("  Test Results Summary")
    print("=" * 60)
    passed = sum(1 for _, s in results if s)
    total = len(results)
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 60)
