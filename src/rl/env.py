from typing import Optional, Tuple, Dict, Any, Set
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import time
import cv2

from src.core.detector import DinoDetector, Detection
from src.core.screen import capture_screenshot
from src.core.keyboard import KeyboardController
from src.utils.visualization import draw_key_indicators, draw_detections


SURVIVAL_REWARD = 0.1
OBSTACLE_PASS_REWARD = 0.0
GAME_OVER_PENALTY = -1.0
DINO_X_POSITION = 40
MAX_SPEED = 1500.0


class ObstacleTracker:
    def __init__(self):
        self.next_id = 0
        self.active_obstacles: Dict[int, dict] = {}
        self.passed_obstacles: Set[int] = set()
    
    def reset(self):
        self.next_id = 0
        self.active_obstacles.clear()
        self.passed_obstacles.clear()
    
    def update(self, obstacles: list) -> Set[int]:
        new_passed = set()
        current_positions = {}
        
        for obs in obstacles:
            matched_id = self._match_obstacle(obs)
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
            
            current_positions[matched_id] = {
                'x_left': obs.x_left,
                'y_center': obs.y_center,
                'label': obs.label,
                'width': obs.width,
                'height': obs.height
            }
            
            if matched_id in self.active_obstacles:
                prev_x = self.active_obstacles[matched_id]['x_left']
                if prev_x > DINO_X_POSITION and obs.x_left <= DINO_X_POSITION:
                    if matched_id not in self.passed_obstacles:
                        self.passed_obstacles.add(matched_id)
                        new_passed.add(matched_id)
        
        self.active_obstacles = current_positions
        return new_passed
    
    def _match_obstacle(self, obs: Detection) -> Optional[int]:
        best_id = None
        best_dist = float('inf')
        
        for obs_id, data in self.active_obstacles.items():
            if data['label'] != obs.label:
                continue
            y_diff = abs(data['y_center'] - obs.y_center)
            if y_diff > 30:
                continue
            x_diff = abs(data['x_left'] - obs.x_left)
            if x_diff < best_dist and x_diff < 100:
                best_dist = x_diff
                best_id = obs_id
        
        return best_id


class SpeedEstimator:
    def __init__(self, max_history: int = 5):
        self.prev_obstacles = []
        self.prev_time = None
        self.speed_samples: deque = deque(maxlen=max_history)
        self.current_speed = 400.0
    
    def reset(self):
        self.prev_obstacles = []
        self.prev_time = None
        self.speed_samples.clear()
        self.current_speed = 400.0
    
    def update(self, obstacles: list, current_time: float) -> float:
        current_obs = [(obs.x_left, obs.y_center, obs.label) for obs in obstacles]
        
        if self.prev_time is not None and self.prev_obstacles:
            dt = current_time - self.prev_time
            if dt > 0:
                speeds = []
                for curr_x, curr_y, curr_label in current_obs:
                    best_prev = None
                    best_x_diff = float('inf')
                    
                    for prev_x, prev_y, prev_label in self.prev_obstacles:
                        if curr_label != prev_label:
                            continue
                        if abs(curr_y - prev_y) > 30:
                            continue
                        if prev_x > curr_x:
                            x_diff = prev_x - curr_x
                            if x_diff < best_x_diff:
                                best_x_diff = x_diff
                                best_prev = (prev_x, prev_y, prev_label)
                    
                    if best_prev is not None:
                        dx = best_prev[0] - curr_x
                        speeds.append(dx / dt)
                
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    avg_speed = max(300, min(MAX_SPEED, avg_speed))
                    self.speed_samples.append(avg_speed)
                    if self.speed_samples:
                        self.current_speed = sum(self.speed_samples) / len(self.speed_samples)
        
        self.prev_obstacles = current_obs
        self.prev_time = current_time
        return self.current_speed


class DinoGameEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode: Optional[str] = None, only_up: bool = False):
        super().__init__()
        
        self.only_up = only_up
        self.n_actions = 2 if only_up else 3
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.n_actions)
        
        self.detector = DinoDetector()
        self.keyboard = KeyboardController()
        self.obstacle_tracker = ObstacleTracker()
        self.speed_estimator = SpeedEstimator()
        
        self.render_mode = render_mode
        self.image_width = 1129
        self.image_height = 232
        
        self._last_action = 0
        self._step_count = 0
        self._current_image = None
        self._last_result = None
        self._window_name = "RL Training"
        self._window_created = False
        self._total_reward = 0.0
        self._episode_count = 0
    
    def _build_state(self, dino_y: Optional[float], obstacles: list, speed: float) -> np.ndarray:
        state = np.full(12, -1.0, dtype=np.float32)
        
        if dino_y is not None:
            state[0] = dino_y / self.image_height
        else:
            state[0] = 0.5
        
        obstacles_right = [obs for obs in obstacles if obs.x_left > DINO_X_POSITION]
        obstacles_right.sort(key=lambda x: x.x_left)
        
        if len(obstacles_right) >= 1:
            obs = obstacles_right[0]
            state[1] = 0.0 if obs.label == 'cactus' else 1.0
            state[2] = obs.x_left / self.image_width
            state[3] = obs.y_center / self.image_height
            state[4] = obs.width / self.image_width
            state[5] = obs.height / self.image_height
        
        if len(obstacles_right) >= 2:
            obs = obstacles_right[1]
            state[6] = 0.0 if obs.label == 'cactus' else 1.0
            state[7] = obs.x_left / self.image_width
            state[8] = obs.y_center / self.image_height
            state[9] = obs.width / self.image_width
            state[10] = obs.height / self.image_height
        
        state[11] = speed / MAX_SPEED
        
        return state
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.obstacle_tracker.reset()
        self.speed_estimator.reset()
        self._last_action = 0
        self._step_count = 0
        self._total_reward = 0.0
        
        self.keyboard.release_all()
        
        for _ in range(5):
            image = capture_screenshot()
            if image is not None:
                self.image_height, self.image_width = image.shape[:2]
                break
            time.sleep(0.1)
        
        image = capture_screenshot()
        if image is None:
            return self._build_state(None, [], 400.0), {}
        
        self._current_image = image
        result = self.detector.detect(image)
        self._last_result = result
        
        if result.has_restart:
            self.keyboard.press_enter()
            time.sleep(0.5)
        
        dino_y = result.dino.y_center if result.dino else None
        state = self._build_state(dino_y, result.obstacles, 400.0)
        
        if self.render_mode == "human":
            self._render_frame()
        
        return state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        self._last_action = action
        
        self.keyboard.update()
        self.keyboard.execute_action(action)
        
        time.sleep(0.02)
        
        image = capture_screenshot()
        if image is None:
            return self._build_state(None, [], self.speed_estimator.current_speed), SURVIVAL_REWARD, False, False, {}
        
        self._current_image = image
        result = self.detector.detect(image)
        self._last_result = result
        current_time = time.time()
        
        speed = self.speed_estimator.update(result.obstacles, current_time)
        
        new_passed = self.obstacle_tracker.update(result.obstacles)
        
        reward = SURVIVAL_REWARD
        reward += len(new_passed) * OBSTACLE_PASS_REWARD
        self._total_reward += reward
        
        terminated = False
        truncated = False
        
        if result.has_restart:
            reward += GAME_OVER_PENALTY
            terminated = True
            self._episode_count += 1
        
        dino_y = result.dino.y_center if result.dino else None
        state = self._build_state(dino_y, result.obstacles, speed)
        
        info = {
            "speed": speed,
            "passed_obstacles": len(self.obstacle_tracker.passed_obstacles),
            "step": self._step_count,
            "total_reward": self._total_reward,
            "episode_count": self._episode_count
        }
        
        if terminated:
            info["episode"] = {
                "r": self._total_reward,
                "l": self._step_count,
                "t": time.time()
            }
        
        if self.render_mode == "human":
            self._render_frame()
        
        return state, reward, terminated, truncated, info
    
    def _render_frame(self):
        if self._current_image is None:
            return
        
        if self._last_result is not None:
            display_img = draw_detections(self._current_image, self._last_result)
        else:
            display_img = self._current_image.copy()
        
        action_names = ["NOOP", "UP"] if self.only_up else ["NOOP", "UP", "DOWN"]
        action_text = f"Action: {action_names[self._last_action]}"
        cv2.putText(
            display_img,
            action_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_img,
            action_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        
        reward_text = f"Reward: {self._total_reward:.2f}"
        cv2.putText(
            display_img,
            reward_text,
            (display_img.shape[1] - 180, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_img,
            reward_text,
            (display_img.shape[1] - 180, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        
        passed_text = f"Passed: {len(self.obstacle_tracker.passed_obstacles)}"
        cv2.putText(
            display_img,
            passed_text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_img,
            passed_text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        
        episode_text = f"Episode: {self._episode_count}"
        cv2.putText(
            display_img,
            episode_text,
            (display_img.shape[1] - 180, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_img,
            episode_text,
            (display_img.shape[1] - 180, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        
        cv2.imshow(self._window_name, display_img)
        self._window_created = True
        cv2.waitKey(1)
    
    def render(self):
        if self.render_mode == "human":
            self._render_frame()
    
    def close(self):
        self.keyboard.release_all()
        if self._window_created:
            cv2.destroyWindow(self._window_name)
            self._window_created = False
