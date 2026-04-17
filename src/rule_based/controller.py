import time
from collections import deque
from typing import Optional, List
import numpy as np
import supervision as sv

from src.core.detector import Detection


DEBUG = True

TIME_TO_PEAK = 0.18
DINO_X_POSITION = 40
MIN_SPEED = 300
MAX_SPEED = 1500
MIN_TRIGGER_DISTANCE = 0
MAX_TRIGGER_DISTANCE = 600
JUMP_WINDOW_MARGIN = 0


class GameController:
    def __init__(self):
        self.obstacle_history = deque(maxlen=4)
        self.current_speed = 400
        self.speed_samples = []
        self.max_speed_samples = 5

    def _match_obstacle(self, obstacle: Detection, prev_obstacles: List[dict]) -> Optional[dict]:
        class_name = obstacle.label
        y_centroid = obstacle.y_center
        x_left = obstacle.x_left
        
        best_match = None
        best_x_diff = float('inf')
        
        for prev in prev_obstacles:
            if prev['class_name'] != class_name:
                continue
            y_diff = abs(prev['y_centroid'] - y_centroid)
            if y_diff > 30:
                continue
            if prev['x_left'] > x_left:
                x_diff = prev['x_left'] - x_left
                if x_diff < best_x_diff:
                    best_x_diff = x_diff
                    best_match = prev
        
        return best_match

    def _calculate_speed(self, detections: sv.Detections, current_time: float):
        current_obstacles = []
        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i]
            class_name = detections.data['class_name'][i]
            if class_name in ['cactus', 'bird']:
                current_obstacles.append({
                    'class_name': class_name,
                    'x_left': box[0],
                    'y_centroid': (box[1] + box[3]) / 2,
                    'box': box
                })

        if len(self.obstacle_history) >= 3:
            prev_obstacles, prev_time = self.obstacle_history[-3]
            dt = current_time - prev_time
            if dt > 0:
                speeds = []
                for curr in current_obstacles:
                    prev = self._match_obstacle_simple(curr, prev_obstacles)
                    if prev:
                        dx = prev['x_left'] - curr['x_left']
                        if dx > 0:
                            speed = dx / dt
                            speeds.append(speed)
                
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    avg_speed = max(MIN_SPEED, min(MAX_SPEED, avg_speed))
                    self.speed_samples.append(avg_speed)
                    if len(self.speed_samples) > self.max_speed_samples:
                        self.speed_samples.pop(0)
                    self.current_speed = sum(self.speed_samples) / len(self.speed_samples)

        self.obstacle_history.append((current_obstacles, current_time))

    def _match_obstacle_simple(self, curr: dict, prev_obstacles: List[dict]) -> Optional[dict]:
        for prev in prev_obstacles:
            if prev['class_name'] == curr['class_name']:
                y_diff = abs(prev['y_centroid'] - curr['y_centroid'])
                if y_diff < 30 and prev['x_left'] > curr['x_left']:
                    return prev
        return None

    def _get_trigger_distance_for_obstacle(self, obstacle_width: float) -> float:
        base_distance = self.current_speed * TIME_TO_PEAK
        distance = base_distance
        distance = max(MIN_TRIGGER_DISTANCE, min(MAX_TRIGGER_DISTANCE, distance))
        return distance

    def get_action(self, detections: sv.Detections, current_time: Optional[float] = None) -> Optional[str]:
        if current_time is None:
            current_time = time.time()

        self._calculate_speed(detections, current_time)

        closest_cactus = None
        closest_cactus_dist = float('inf')
        closest_bird = None
        closest_bird_dist = float('inf')

        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i]
            y_centroid = (box[1] + box[3]) / 2
            class_name = detections.data['class_name'][i]
            obstacle_width = box[2] - box[0]
            distance_to_dino = box[0] - DINO_X_POSITION

            if DEBUG:
                print(f"[{class_name}] x_left={box[0]:.1f}, x_right={box[2]:.1f}, "
                      f"width={obstacle_width:.1f}, y_centroid={y_centroid:.1f}, "
                      f"distance={distance_to_dino:.1f}")

            if class_name == 'cactus':
                y_in_range = 100 < y_centroid < 190
                if y_in_range and 0 < distance_to_dino < closest_cactus_dist:
                    closest_cactus = {
                        'box': box,
                        'distance': distance_to_dino,
                        'width': obstacle_width,
                        'y_centroid': y_centroid
                    }
                    closest_cactus_dist = distance_to_dino

            elif class_name == 'bird':
                if 0 < distance_to_dino < closest_bird_dist:
                    closest_bird = {
                        'box': box,
                        'distance': distance_to_dino,
                        'width': obstacle_width,
                        'y_centroid': y_centroid
                    }
                    closest_bird_dist = distance_to_dino

        if closest_cactus:
            dist = closest_cactus['distance']
            width = closest_cactus['width']
            trigger_distance = self._get_trigger_distance_for_obstacle(width)
            
            jump_max = trigger_distance + JUMP_WINDOW_MARGIN
            should_jump = MIN_TRIGGER_DISTANCE < dist <= jump_max
            
            if DEBUG:
                print(f"  -> cactus: dist={dist:.1f}, width={width:.1f}, "
                      f"trigger={trigger_distance:.1f}, range=[{MIN_TRIGGER_DISTANCE},{jump_max:.0f}], "
                      f"jump={should_jump}")
            
            if should_jump:
                return "up"

        if closest_bird:
            dist = closest_bird['distance']
            width = closest_bird['width']
            y_centroid = closest_bird['y_centroid']
            trigger_distance = self._get_trigger_distance_for_obstacle(width)
            
            bird_max = trigger_distance + JUMP_WINDOW_MARGIN + 30
            in_range = MIN_TRIGGER_DISTANCE < dist <= bird_max
            
            if DEBUG:
                print(f"  -> bird: dist={dist:.1f}, width={width:.1f}, "
                      f"y_centroid={y_centroid:.1f}, trigger={trigger_distance:.1f}, "
                      f"range=[{MIN_TRIGGER_DISTANCE},{bird_max:.0f}], in_range={in_range}")
            
            if in_range:
                if y_centroid > 150:
                    return "up"
                else:
                    return "down"

        return None

    def get_current_speed(self) -> float:
        return self.current_speed
