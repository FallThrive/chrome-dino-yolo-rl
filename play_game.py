from ultralytics import YOLO
from take_screenshots import capture_screenshot
import supervision as sv
import cv2
import time
import numpy as np
from pynput.keyboard import Key, Controller

from controller import get_action
from controller import get_current_speed
from utils import draw_key_indicators


model = YOLO("weights/yolo26n_dino_260418.pt")

palette = sv.ColorPalette(colors=[sv.Color(112, 153, 223), sv.Color(124, 221, 161), sv.Color(163, 81, 251)])

bounding_box_annotator = sv.BoxAnnotator(color=palette)
label_annotator = sv.LabelAnnotator(color=palette)

keyboard = Controller()

keyboard_img = None

key_release_queue = {}

game_active = True

def has_label(detections, label_name):
    if len(detections) == 0:
        return False
    class_names = detections.data.get('class_name', [])
    return label_name in class_names

while True:
    current_time = time.time()
    keys_to_release = []
    for key, release_time in key_release_queue.items():
        if current_time >= release_time:
            keyboard.release(key)
            keys_to_release.append(key)
    for key in keys_to_release:
        del key_release_queue[key]

    image = capture_screenshot()
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.confidence > 0.6]

    has_restart = has_label(detections, 'restart')
    has_dino = has_label(detections, 'dino')

    if has_restart:
        game_active = False
    elif not has_restart and has_dino:
        game_active = True

    image = bounding_box_annotator.annotate(
        scene=image,
        detections=detections
    )
    image = label_annotator.annotate(
        scene=image,
        detections=detections,
    )

    if game_active:
        action = get_action(detections, current_time)

        if action == "up":
            if Key.space not in key_release_queue:
                keyboard.press(Key.space)
                key_release_queue[Key.space] = time.time() + 0.3
        elif action == "down":
            if Key.down not in key_release_queue:
                keyboard.press(Key.down)
                key_release_queue[Key.down] = time.time() + 0.4

    if keyboard_img is None:
        keyboard_img = np.ones((64 * 2 + 20, image.shape[1], 3), dtype=np.uint8) * 255
    keyboard_img = draw_key_indicators(keyboard_img, key_release_queue.keys())

    speed_text = f"Speed: {get_current_speed():.1f}px/s"
    cv2.putText(
        image,
        speed_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        speed_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    status_text = "PLAYING" if game_active else "WAITING"
    status_color = (0, 255, 0) if game_active else (0, 0, 255)
    cv2.putText(
        image,
        status_text,
        (image.shape[1] - 120, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        status_text,
        (image.shape[1] - 120, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("Gameplay", cv2.vconcat([image, keyboard_img]))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
