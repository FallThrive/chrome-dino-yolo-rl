import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class EpisodeLimitCallback(BaseCallback):
    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        
        for done in dones:
            if done:
                self.episode_count += 1
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}/{self.max_episodes}")
                
                if self.max_episodes > 0 and self.episode_count >= self.max_episodes:
                    if self.verbose > 0:
                        print(f"\nReached {self.max_episodes} episodes. Stopping training.")
                    return False
        
        return True


class DinoTrainingCallback(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -float('inf')
        self._last_done = False
    
    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "best"), exist_ok=True)
    
    def _on_step(self) -> bool:
        try:
            import keyboard
            if keyboard.is_pressed('q'):
                if self.verbose > 0:
                    print("Training stopped by user (Q key pressed)")
                self.model.save(os.path.join(self.save_path, "final_model.zip"))
                return False
        except ImportError:
            pass
        
        return True
    
    def _on_rollout_end(self) -> None:
        try:
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                ep_rewards = []
                for ep_info in self.model.ep_info_buffer:
                    if isinstance(ep_info, dict) and "r" in ep_info:
                        ep_rewards.append(ep_info["r"])
                
                if ep_rewards:
                    mean_reward = np.mean(ep_rewards)
                    
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        best_path = os.path.join(self.save_path, "best", "model.zip")
                        self.model.save(best_path)
                        if self.verbose > 0:
                            print(f"New best model saved! Mean reward: {mean_reward:.2f}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not process ep_info_buffer: {e}")


class GameResetCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        for done in dones:
            if done:
                from src.core.keyboard import KeyboardController
                keyboard = KeyboardController()
                keyboard.press_enter()
                break
        
        return True
