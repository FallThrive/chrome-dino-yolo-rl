import os
import sys
import cv2
import numpy as np
import time
from pynput.keyboard import Key

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.detector import DinoDetector
from src.core.screen import capture_screenshot
from src.core.keyboard import KeyboardController
from src.utils.visualization import draw_key_indicators, draw_detections, FPSCounter, draw_fps
from src.rule_based.controller import GameController


def play_rule_based():
    detector = DinoDetector()
    keyboard = KeyboardController()
    controller = GameController()
    fps_counter = FPSCounter()
    
    keyboard_img = None
    game_active = True
    
    print("Starting rule-based gameplay...")
    print("Press 'q' to quit")
    
    while True:
        keyboard.update()
        
        image = capture_screenshot()
        if image is None:
            continue
        
        result = detector.detect(image)
        detections = result.raw_detections
        
        has_restart = result.has_restart
        has_dino = result.dino is not None

        if has_restart:
            game_active = False
        elif not has_restart and has_dino:
            game_active = True

        if game_active:
            action = controller.get_action(detections, time.time())

            if action == "up":
                keyboard.press_jump()
            elif action == "down":
                keyboard.press_duck()

        display_img = draw_detections(image, result)
        
        fps_counter.update()
        draw_fps(display_img, fps_counter.fps)

        if keyboard_img is None:
            keyboard_img = np.ones((64 * 2 + 20, image.shape[1], 3), dtype=np.uint8) * 255
        keyboard_img = draw_key_indicators(keyboard_img, keyboard.get_pressed_keys())

        speed_text = f"Speed: {controller.get_current_speed():.1f}px/s"
        cv2.putText(
            display_img,
            speed_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_img,
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
            display_img,
            status_text,
            (display_img.shape[1] - 120, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_img,
            status_text,
            (display_img.shape[1] - 120, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Gameplay", cv2.vconcat([display_img, keyboard_img]))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    play_rule_based()
