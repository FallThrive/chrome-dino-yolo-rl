import os
import numpy as np
import time
from stable_baselines3.common.callbacks import BaseCallback


class TrainingStatsCallback(BaseCallback):
    def __init__(self, save_path: str = "", print_freq: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.print_freq = print_freq
        self.episode_count = 0
        self.last_print_time = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -float('inf')
        self.new_best_saved = False
    
    def _init_callback(self) -> None:
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "best"), exist_ok=True)
    
    def _on_step(self) -> bool:
        try:
            import keyboard
            if keyboard.is_pressed('q'):
                if self.save_path:
                    self.model.save(os.path.join(self.save_path, "final_model.zip"))
                return False
        except ImportError:
            pass
        
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                if i < len(infos) and 'episode' in infos[i]:
                    self.episode_rewards.append(infos[i]['episode']['r'])
                    self.episode_lengths.append(infos[i]['episode']['l'])
        
        current_time = time.time()
        if self.verbose > 0 and current_time - self.last_print_time >= self.print_freq:
            self._print_stats()
            self.last_print_time = current_time
        
        return True
    
    def _print_stats(self):
        if len(self.episode_rewards) == 0:
            return
        
        mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
        
        last_reward = self.episode_rewards[-1] if self.episode_rewards else 0
        last_length = self.episode_lengths[-1] if self.episode_lengths else 0
        
        timesteps = self.num_timesteps
        
        if mean_reward > self.best_mean_reward and self.save_path:
            self.best_mean_reward = mean_reward
            self.new_best_saved = True
            best_path = os.path.join(self.save_path, "best", "model.zip")
            self.model.save(best_path)
        else:
            self.new_best_saved = False
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 30)
        print("   Chrome Dino RL Training Progress")
        print("=" * 30)
        print(f"  Timesteps:     {timesteps:,}")
        print(f"  Episodes:      {self.episode_count}")
        print("-" * 30)
        print(f"  Last Episode:")
        print(f"    - Reward:    {last_reward:.2f}")
        print(f"    - Length:    {last_length:.0f} steps")
        print("-" * 30)
        print(f"  Mean (last 100 episodes):")
        print(f"    - Reward:    {mean_reward:.2f}")
        print(f"    - Length:    {mean_length:.0f} steps")
        print(f"    - Best:      {self.best_mean_reward:.2f}")
        if self.new_best_saved:
            print("-" * 30)
            print("  >>> NEW BEST MODEL SAVED! <<<")
        print("=" * 30)
        print("  Press 'Q' to stop training")
        print("=" * 30)


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
