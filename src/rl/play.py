import argparse
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO

from src.core.keyboard import KeyboardController
from src.utils.visualization import draw_key_indicators, draw_detections, FPSCounter, draw_fps
from src.rl.env import DinoGameEnv


def find_latest_model(base_path: str = "weights/rl") -> str:
    """Find the latest model in the weights directory."""
    if not os.path.exists(base_path):
        return None
    
    subdirs = [d for d in os.listdir(base_path) 
               if os.path.isdir(os.path.join(base_path, d)) and d != "best" and d != "checkpoints"]
    
    if not subdirs:
        best_path = os.path.join(base_path, "best", "model.zip")
        if os.path.exists(best_path):
            return best_path
        return None
    
    subdirs.sort(reverse=True)
    
    for subdir in subdirs:
        best_path = os.path.join(base_path, subdir, "best", "model.zip")
        if os.path.exists(best_path):
            return best_path
        
        final_path = os.path.join(base_path, subdir, "final_model.zip")
        if os.path.exists(final_path):
            return final_path
        
        checkpoint_dir = os.path.join(base_path, subdir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
            if checkpoints:
                checkpoints.sort(reverse=True)
                return os.path.join(checkpoint_dir, checkpoints[0])
    
    return None


def play_rl(weights_path: str = None, use_latest: bool = False, only_up: bool = False):
    if use_latest or weights_path is None:
        weights_path = find_latest_model()
        if weights_path is None:
            print("No trained model found. Please train a model first.")
            return
        print(f"Using latest model: {weights_path}")
    elif not weights_path.endswith('.zip'):
        weights_path += '.zip'
    
    env = DinoGameEnv(render_mode=None, only_up=only_up)
    
    try:
        model = PPO.load(weights_path, env=env)
        print(f"Loaded model from {weights_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training a new model...")
        model = PPO("MlpPolicy", env, verbose=1)
    
    keyboard = KeyboardController()
    fps_counter = FPSCounter()
    
    keyboard_img = None
    
    print("=" * 50)
    print("Playing Chrome Dino Game with RL Agent")
    print("=" * 50)
    print(f"Model: {weights_path}")
    print(f"Action space: {'UP only' if only_up else 'UP/DOWN'}")
    print("Press 'Q' to quit")
    print("=" * 50)
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    episode_count = 0
    
    while True:
        keyboard.update()
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if done or truncated:
            episode_count += 1
            print(f"Episode {episode_count} ended. Total reward: {total_reward:.2f}, Steps: {step_count}")
            keyboard.press_enter()
            time.sleep(0.5)
            obs, _ = env.reset()
            fps_counter.reset()
            total_reward = 0
            step_count = 0
            done = False
        
        if env._current_image is not None and env._last_result is not None:
            image = env._current_image
            result = env._last_result
            
            display_img = draw_detections(image, result)
            
            fps_counter.update()
            draw_fps(display_img, fps_counter.fps)
            
            if keyboard_img is None:
                keyboard_img = np.ones((64 * 2 + 20, image.shape[1], 3), dtype=np.uint8) * 255
            keyboard_img = draw_key_indicators(keyboard_img, keyboard.get_pressed_keys())
            
            action_names = ["NOOP", "UP"] if only_up else ["NOOP", "UP", "DOWN"]
            action_text = f"Action: {action_names[action]}"
            cv2.putText(
                display_img,
                action_text,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_img,
                action_text,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            
            reward_text = f"Reward: {total_reward:.2f}"
            cv2.putText(
                display_img,
                reward_text,
                (display_img.shape[1] - 180, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_img,
                reward_text,
                (display_img.shape[1] - 180, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            
            cv2.imshow("RL Gameplay", cv2.vconcat([display_img, keyboard_img]))
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    
    cv2.destroyAllWindows()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Play Chrome Dino Game with trained RL agent")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--latest", action="store_true", help="Use the latest trained model")
    parser.add_argument("--only-up", action="store_true", help="Use only UP action (no DOWN)")
    
    args = parser.parse_args()
    
    play_rl(weights_path=args.weights, use_latest=args.latest, only_up=args.only_up)


if __name__ == "__main__":
    main()
