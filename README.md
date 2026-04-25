# Chrome Dino YOLO RL

English | [简体中文](README_CN.md)

A Chrome Dino Game bot using YOLO26 object detection model and Reinforcement Learning (PPO algorithm). The project supports two control methods:

1. **Rule-based Control**: Uses predefined rules for game decisions
2. **Reinforcement Learning Control**: Uses PPO (Proximal Policy Optimization) algorithm to train an intelligent agent

## Tech Stack

- **Python 3.12+**
- **YOLO26** - Object detection model (via Ultralytics)
- **Stable Baselines3** - Reinforcement learning framework (PPO algorithm)
- **Gymnasium** - Reinforcement learning environment interface
- **OpenCV** - Image processing
- **mss** - Screen capture
- **keyboard** - Keyboard control
- **supervision** - Object detection visualization

## Environment Setup

This project uses `uv` as the package manager.

### Clone the Repository

```bash
git clone https://github.com/FallThrive/chrome-dino-yolo-rl.git
cd chrome-dino-yolo-rl
```

### Install Dependencies

```bash
uv sync
```

## Quick Start

### Method 1: Rule-based Control

#### Step 1: Start Chrome Dino Game
1. Open Chrome browser
2. Visit `chrome://dino/` or visit any webpage after disconnecting from the network
3. Press spacebar to start the game and let the dino run

#### Step 2: Run Rule-based Control Program

```bash
uv run python src/rule_based/play.py
```

On first run, you will be prompted to:
1. **Select Monitor**: Enter monitor number (usually 0)
2. **Select Game Area**: Use mouse to select the dino game area, press **ENTER** to confirm

> Configuration will be automatically saved to `config.json`, no need to reselect next time

#### Step 3: Observe Running Results
- The program will display the game window with detection boxes and labels
- Current key press status indicator is displayed at the bottom
- Press **Q** to exit the program

### Method 2: Reinforcement Learning Control

#### Train Model

```bash
uv run python src/rl/train.py
```

Training parameters (can be modified via command line arguments):
- `--timesteps`: Total training steps (default: 1000000)
- `--episodes`: Maximum training episodes (default: 0, no limit)
- `--lr`: Learning rate (default: 3e-4)
- `--save-freq`: Checkpoint save frequency (default: 10000)
- `--save-path`: Model save path (default: weights/rl)
- `--log-dir`: TensorBoard log directory (default: logs)
- `--no-render`: Disable visualization window during training (default: enabled)
- `--only-up`: Use only jump action (no duck, action space becomes 2)

During training:
- Model and log directories automatically add timestamps (format: `YYMMDD_HHMMSS`)
- If `--episodes` is specified, training stops automatically after reaching the specified number of episodes
- Press **Q** to stop training at any time and save the current model
- Models are automatically saved to `weights/rl/<timestamp>/checkpoints/`
- Best model saved to `weights/rl/<timestamp>/best/model.zip`
- Final model saved to `weights/rl/<timestamp>/final_model.zip`

#### Play Game with Trained Model

```bash
# Use the latest trained model
uv run python src/rl/play.py --latest

# Specify model path
uv run python src/rl/play.py --weights weights/rl/250418_143052/best/model.zip
```

#### View Training Curves

```bash
tensorboard --logdir logs/
```

### Data Collection

Used to collect game screenshots for training YOLO model:

```bash
uv run python src/core/take_screenshots.py
```

After running:
1. Prompt to select monitor
2. Select game area ROI
3. Automatically capture screenshots in a loop and save to `dataset/` directory

## Project Structure

```
chrome-dino-yolo-rl/
├── src/
│   ├── core/                    # Core shared modules
│   │   ├── __init__.py
│   │   ├── detector.py          # YOLO detector wrapper
│   │   ├── keyboard.py          # Keyboard control
│   │   ├── screen.py            # Screen capture
│   │   └── take_screenshots.py  # Data collection script
│   │
│   ├── rule_based/              # Rule-based control method
│   │   ├── __init__.py
│   │   ├── controller.py        # Rule controller
│   │   └── play.py              # Game entry point
│   │
│   ├── rl/                      # Reinforcement learning module
│   │   ├── __init__.py
│   │   ├── callbacks.py         # SB3 callbacks
│   │   ├── env.py               # Gymnasium environment wrapper
│   │   ├── play.py              # Play with trained model
│   │   └── train.py             # Training script
│   │
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   └── visualization.py     # Visualization tools
│   │
│   ├── yolo/                    # YOLO training scripts
│   │   ├── train_yolo26n.py     # Standard training script
│   │   └── train_yolo26n_simple.py  # Simplified training script
│   │
│   └── __init__.py
│
├── weights/                     # Model weights directory
│   ├── yolo26n_dino.pt          # YOLO detection model
│   ├── yolo26n_dino_simple.pt   # YOLO simplified detection model
│   └── rl/                      # RL model weights
│       └── <timestamp>/         # Training timestamp directory (YYMMDD_HHMMSS)
│           ├── best/            # Best weights
│           ├── checkpoints/     # Checkpoints
│           └── final_model.zip  # Final model
│
├── logs/                        # TensorBoard logs
│   └── <timestamp>/             # Training timestamp directory (YYMMDD_HHMMSS)
├── cfg/                         # Configuration files
│   ├── roi.json                 # ROI configuration
│   └── yolo26n_simple.yaml      # YOLO model configuration
├── dataset/                     # Training dataset directory
├── runs/                        # YOLO training results
├── assets/                      # UI icon resources
├── config.json                  # ROI configuration (generated at runtime)
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependencies
└── README.md                    # This document
```

## YOLO26 Model

Uses locally trained YOLO26 model:
- **Original Network Model**: `weights/yolo26n_dino.pt`
- **Simplified Network Model**: `weights/yolo26n_dino_simple.pt`
- **Detection Classes**: `dino` (dinosaur), `cactus` (cactus), `bird` (bird), `restart` (restart)

## Configuration

### config.json (Generated at Runtime)

```json
{
    "top": 100,
    "left": 200,
    "width": 600,
    "height": 150,
    "mon": 1
}
```

## Dependencies

| Package               | Purpose                    |
|-----------------------|----------------------------|
| ultralytics           | YOLO26 model inference     |
| stable-baselines3     | PPO reinforcement learning |
| gymnasium             | RL environment interface   |
| keyboard              | Keyboard event simulation  |
| supervision           | Object detection visualization |
| opencv-contrib-python | Image processing           |
| mss                   | Screen capture             |
| numpy                 | Numerical computation      |
| torch                 | Deep learning framework    |
| torchvision           | PyTorch vision library     |

## FAQ

### Q: How to reselect the game area?
Delete the `config.json` file and run the program again.

### Q: Detection is not accurate?
- Ensure the game area is selected correctly
- Adjust the confidence threshold in `src/core/detector.py` (default: 0.6)
- Collect more data to train a custom YOLO model

### Q: Training not converging?
- Adjust learning rate (try 1e-4 to 1e-3)
- Increase training steps
- Check if reward function design is reasonable

### Q: GPU acceleration not working?
Ensure CUDA and cuDNN are installed, PyTorch will automatically use GPU.

## Extension Development Suggestions

1. **Improve Reward Function**: Adjust reward parameters or add new reward items
2. **Custom Network Architecture**: Modify `policy_kwargs` in `src/rl/train.py`
3. **Multi-game Support**: Extend `src/rl/env.py` to support other games
4. **Data Collection**: Use `src/core/take_screenshots.py` to collect more training data

---

## Acknowledgements

This project is refactored based on [Erol444/chrome-dino-bot](https://github.com/Erol444/chrome-dino-bot).
