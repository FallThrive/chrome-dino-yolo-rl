import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.env import DinoGameEnv
from src.rl.callbacks import DinoTrainingCallback, GameResetCallback, EpisodeLimitCallback


def train(
    total_timesteps: int = 1000000,
    n_episodes: int = 0,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    save_freq: int = 10000,
    save_path: str = "weights/rl",
    log_dir: str = "logs",
    continue_from: str = None,
    render: bool = True,
    only_up: bool = False,
):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    save_path = os.path.join(save_path, timestamp)
    log_dir = os.path.join(log_dir, timestamp)
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "best"), exist_ok=True)
    
    render_mode = "human" if render else None
    env = DummyVecEnv([lambda: DinoGameEnv(render_mode=render_mode, only_up=only_up)])
    
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )
    
    if continue_from and os.path.exists(continue_from):
        print(f"Loading model from {continue_from}")
        model = PPO.load(continue_from, env=env, device='cpu')
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='cpu',
        )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix="dino_ppo"
    )
    
    dino_callback = DinoTrainingCallback(save_path, verbose=1)
    reset_callback = GameResetCallback(verbose=0)
    episode_callback = EpisodeLimitCallback(n_episodes, verbose=1)
    
    callbacks = [checkpoint_callback, dino_callback, reset_callback, episode_callback]
    
    print("=" * 50)
    print("Starting PPO Training for Chrome Dino Game")
    print("=" * 50)
    print(f"Timestamp: {timestamp}")
    print(f"Total timesteps: {total_timesteps}")
    if n_episodes > 0:
        print(f"Max episodes: {n_episodes}")
    print(f"Action space: {'UP only (2 actions)' if only_up else 'UP/DOWN (3 actions)'}")
    print(f"Learning rate: {learning_rate}")
    print(f"Save frequency: {save_freq}")
    print(f"Save path: {save_path}")
    print(f"Log directory: {log_dir}")
    print("=" * 50)
    print("Press 'Q' to stop training and save the model")
    print("=" * 50)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        final_path = os.path.join(save_path, "final_model.zip")
        model.save(final_path)
        print(f"Final model saved to {final_path}")
    
    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Chrome Dino Game")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--episodes", type=int, default=0, help="Max training episodes (0 = unlimited)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-freq", type=int, default=10000, help="Checkpoint save frequency")
    parser.add_argument("--save-path", type=str, default="weights/rl", help="Model save path")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--continue", dest="continue_from", type=str, default=None, help="Continue training from model path")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering during training")
    parser.add_argument("--only-up", action="store_true", help="Use only UP action (no DOWN)")
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        n_episodes=args.episodes,
        learning_rate=args.lr,
        save_freq=args.save_freq,
        save_path=args.save_path,
        log_dir=args.log_dir,
        continue_from=args.continue_from,
        render=not args.no_render,
        only_up=args.only_up,
    )


if __name__ == "__main__":
    main()
