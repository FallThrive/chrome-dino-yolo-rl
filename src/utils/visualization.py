import cv2
import numpy as np
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.detector import DinoDetectionResult

ASSETS_DIR = "assets"


class FPSCounter:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: list = []
        self._last_fps = 0.0
    
    def update(self) -> float:
        current_time = time.time()
        self.timestamps.append(current_time)
        
        while len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        if len(self.timestamps) >= 2:
            time_span = self.timestamps[-1] - self.timestamps[0]
            if time_span > 0:
                self._last_fps = (len(self.timestamps) - 1) / time_span
        
        return self._last_fps
    
    @property
    def fps(self) -> float:
        return self._last_fps
    
    def reset(self):
        self.timestamps.clear()
        self._last_fps = 0.0


def draw_fps(image: np.ndarray, fps: float, position: tuple = None) -> np.ndarray:
    if position is None:
        position = (image.shape[1] // 2 - 40, 28)
    
    fps_text = f"FPS: {fps:.1f}"
    
    cv2.putText(
        image,
        fps_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        fps_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    
    return image


def draw_detections(image: np.ndarray, result: "DinoDetectionResult") -> np.ndarray:
    display_img = image.copy()
    
    if result.dino is not None:
        dino = result.dino
        cv2.rectangle(
            display_img,
            (int(dino.x_left), int(dino.y_top)),
            (int(dino.x_right), int(dino.y_bottom)),
            (0, 255, 0),
            2
        )
        cv2.putText(
            display_img,
            f"dino {dino.confidence:.2f}",
            (int(dino.x_left), int(dino.y_top) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    
    for obs in result.obstacles:
        if obs.label == 'cactus':
            color = (0, 0, 255)
        elif obs.label == 'bird':
            color = (255, 0, 0)
        else:
            color = (128, 128, 128)
        
        cv2.rectangle(
            display_img,
            (int(obs.x_left), int(obs.y_top)),
            (int(obs.x_right), int(obs.y_bottom)),
            color,
            2
        )
        cv2.putText(
            display_img,
            f"{obs.label} {obs.confidence:.2f}",
            (int(obs.x_left), int(obs.y_top) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    
    return display_img


def load_icon(filename: str) -> np.ndarray:
    path = os.path.join(ASSETS_DIR, filename)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Warning: Could not read asset: {path}. Using a placeholder.")
        return np.zeros((64, 64, 3), dtype=np.uint8)

    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        rgb = image[:, :, :3]
        white_bg = np.full(rgb.shape, 255, dtype=np.uint8)
        alpha_mask = alpha[:, :, np.newaxis] / 255.0
        composite = rgb * alpha_mask + white_bg * (1 - alpha_mask)
        return composite.astype(np.uint8)
    else:
        return image


UP_IMG = load_icon("up.png")
UP_PRESSED_IMG = load_icon("up-pressed.png")
DOWN_IMG = load_icon("down.png")
DOWN_PRESSED_IMG = load_icon("down-pressed.png")
LEFT_IMG = load_icon("left.png")
RIGHT_IMG = load_icon("right.png")


def overlay_image(background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay.shape

    if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h:
        return background

    background[y:y+h, x:x+w] = overlay
    return background


def draw_key_indicators(image: np.ndarray, pressed_keys) -> np.ndarray:
    icon_h, icon_w, _ = UP_IMG.shape
    screen_h, screen_w, _ = image.shape
    margin = 10

    total_width = 3 * icon_w
    start_x = (screen_w - total_width) // 2

    up_x = start_x + icon_w
    up_y = margin

    down_x = start_x + icon_w
    down_y = margin + icon_h

    left_x = start_x
    left_y = margin + icon_h

    right_x = start_x + (2 * icon_w)
    right_y = margin + icon_h

    up_icon = UP_PRESSED_IMG if 'space' in pressed_keys else UP_IMG
    image = overlay_image(image, up_icon, up_x, up_y)
    
    down_icon = DOWN_PRESSED_IMG if 'down' in pressed_keys else DOWN_IMG
    image = overlay_image(image, down_icon, down_x, down_y)

    image = overlay_image(image, LEFT_IMG, left_x, left_y)
    image = overlay_image(image, RIGHT_IMG, right_x, right_y)

    return image
