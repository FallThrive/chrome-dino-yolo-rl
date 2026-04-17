from typing import Optional, Tuple, Dict, Any, Set
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import time

from ..core.detector import DinoDetector, Detection
from ..core.screen import capture_screenshot
from ..core.keyboard import KeyboardController


SURVIVAL_REWARD = 0.01
OBSTACLE_PASS_REWARD = 1.0
GAME_OVER_PENALTY = -2.0
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
    def __init__(self, max_history: int = 4):
        self.history: deque = deque(maxlen=max_history)
        self.speed_samples: deque = deque(maxlen=5)
        self.current_speed = 400.0
    
    def reset(self):
        self.history.clear()
        self.speed_samples.clear()
        self.current_speed = 400.0
    
    def update(self, obstacles: list, current_time: float) -> float:
        current_obs = [(obs.x_left, obs.y_center, obs.label) for obs in obstacles]
        
        if len(self.history) >= 3:
            prev_obs, prev_time = self.history[-3]
            dt = current_time - prev_time
            if dt > 0:
                speeds = []
                for curr_x, curr_y, curr_label in current_obs:
                    for prev_x, prev_y, prev_label in prev_obs:
                        if curr_label != prev_label:
                            continue
                        if abs(curr_y - prev_y) > 30:
                            continue
                        if prev_x > curr_x:
                            dx = prev_x - curr_x
                            speeds.append(dx / dt)
                
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    avg_speed = max(300, min(MAX_SPEED, avg_speed))
                    self.speed_samples.append(avg_speed)
                    if self.speed_samples:
                        self.current_speed = sum(self.speed_samples) / len(self.speed_samples)
        
        self.history.append((current_obs, current_time))
        return self.current_speed


class DinoGameEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)
        
        self.detector = DinoDetector()
        self.keyboard = KeyboardController()
        self.obstacle_tracker = ObstacleTracker()
        self.speed_estimator = SpeedEstimator()
        
        self.render_mode = render_mode
        self.image_width = 1129
        self.image_height = 232
        
        self._last_action = 0
        self._step_count = 0
    
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
        
        result = self.detector.detect(image)
        
        if result.has_restart:
            self.keyboard.press_enter()
            time.sleep(0.5)
        
        dino_y = result.dino.y_center if result.dino else None
        state = self._build_state(dino_y, result.obstacles, 400.0)
        
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
        
        result = self.detector.detect(image)
        current_time = time.time()
        
        speed = self.speed_estimator.update(result.obstacles, current_time)
        
        new_passed = self.obstacle_tracker.update(result.obstacles)
        
        reward = SURVIVAL_REWARD
        reward += len(new_passed) * OBSTACLE_PASS_REWARD
        
        terminated = False
        truncated = False
        
        if result.has_restart:
            reward += GAME_OVER_PENALTY
            terminated = True
        
        dino_y = result.dino.y_center if result.dino else None
        state = self._build_state(dino_y, result.obstacles, speed)
        
        info = {
            "speed": speed,
            "passed_obstacles": len(self.obstacle_tracker.passed_obstacles),
            "step": self._step_count
        }
        
        return state, reward, terminated, truncated, info
    
    def render(self):
        pass
    
    def close(self):
        self.keyboard.release_all()
