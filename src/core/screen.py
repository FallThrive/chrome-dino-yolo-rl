import os
import json
import mss
import numpy as np
import cv2

CONFIG_FILE = "config.json"
DATASET_DIR = "dataset"


def get_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None


def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)


def get_roi_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Error reading {CONFIG_FILE}: {e}")
            return None
    else:
        print(f"Configuration file {CONFIG_FILE} not found. Please run this script first to create it.")
        return None


_monitor_config = None


def capture_screenshot():
    global _monitor_config
    if _monitor_config is None:
        _monitor_config = get_roi_config()
        if _monitor_config is None:
            print("No ROI configuration found. Please select a region.")
            _monitor_config = select_screen_and_roi()
            if _monitor_config is None:
                print("ROI selection is required to capture a screenshot.")
                return None

    with mss.mss() as sct:
        sct_img = np.array(sct.grab(_monitor_config))
        image = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)
        return image


def select_screen_and_roi():
    with mss.mss() as sct:
        monitors = sct.monitors[1:]

        if not monitors:
            print("No external monitors found, using the primary display.")
            monitors = [sct.monitors[1]] if len(sct.monitors) > 1 else []
            if not monitors:
                print("Error: No monitors detected.")
                return None

        print("Please select a monitor to capture:")
        for i, monitor in enumerate(monitors):
            print(f"  {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")

        while True:
            try:
                monitor_number = int(input(f"Enter monitor number (0-{len(monitors)-1}): "))
                if 0 <= monitor_number < len(monitors):
                    break
                else:
                    print("Invalid monitor number.")
            except ValueError:
                print("Please enter a valid number.")

        selected_monitor = monitors[monitor_number]

        sct_img = sct.grab(selected_monitor)
        img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        window_name = "Select ROI - Press ENTER to confirm, C to cancel"
        roi = cv2.selectROI(window_name, img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        if not any(roi):
            print("ROI selection cancelled.")
            return None

        x, y, w, h = roi

        roi_abs = {
            "top": selected_monitor['top'] + y,
            "left": selected_monitor['left'] + x,
            "width": w,
            "height": h,
            "mon": monitor_number + 1
        }

        save_config(roi_abs)
        print(f"Configuration saved to {CONFIG_FILE}")

        return roi_abs


def reset_monitor_config():
    global _monitor_config
    _monitor_config = None


if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    import time
    while True:
        image = capture_screenshot()
        if image is not None:
            cv2.imwrite(f"{DATASET_DIR}/{time.time()}.png", image)
        time.sleep(0.5)
