import cv2
import numpy as np
import os
from pynput.keyboard import Key

ASSETS_DIR = "assets"


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

    up_icon = UP_PRESSED_IMG if Key.space in pressed_keys else UP_IMG
    image = overlay_image(image, up_icon, up_x, up_y)
    
    down_icon = DOWN_PRESSED_IMG if Key.down in pressed_keys else DOWN_IMG
    image = overlay_image(image, down_icon, down_x, down_y)

    image = overlay_image(image, LEFT_IMG, left_x, left_y)
    image = overlay_image(image, RIGHT_IMG, right_x, right_y)

    return image
