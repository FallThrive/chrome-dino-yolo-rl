import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingStatsCallback(BaseCallback):
    def __init__(self, save_path: str = "", verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -float('inf')
    
    def _init_callback(self) -> None:
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "best"), exist_ok=True)
    
    def _on_step(self) -> bool:
        try:
            import keyboard
            if keyboard.is_pressed('q') or keyboard.is_pressed('Q'):
                raise KeyboardInterrupt
        except ImportError:
            pass

        env = self.model.env.envs[0] if hasattr(self.model, 'env') else None
        if env is not None and getattr(env, 'quit_requested', False):
            raise KeyboardInterrupt

        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])

        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                if i < len(infos) and 'episode' in infos[i]:
                    r = infos[i]['episode']['r']
                    l = infos[i]['episode']['l']
                    self.episode_rewards.append(r)
                    self.episode_lengths.append(l)

                    mean_reward = np.mean(self.episode_rewards[-100:])
                    if mean_reward > self.best_mean_reward and self.save_path:
                        self.best_mean_reward = mean_reward
                        best_path = os.path.join(self.save_path, "best", "model.zip")
                        self.model.save(best_path)

        return True


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
