# Chrome Dino Bot - AI Agent 项目指南

## 项目简介

这是一个使用 YOLO26 目标检测模型自动玩 Chrome 恐龙游戏的 AI 机器人项目。项目支持两种控制方式：

1. **基于规则的控制**：使用预设规则进行游戏决策
2. **强化学习控制**：使用 PPO（Proximal Policy Optimization）算法训练智能体

## 技术栈

- **Python 3.12+**
- **YOLO26** - 目标检测模型 (通过 Ultralytics)
- **Stable Baselines3** - 强化学习框架 (PPO算法)
- **Gymnasium** - 强化学习环境接口
- **OpenCV** - 图像处理
- **mss** - 屏幕截图捕获
- **pynput** - 键盘控制
- **supervision** - 目标检测可视化

## 环境配置 (uv)

本项目使用 `uv` 作为包管理器。

### 安装依赖

```bash
uv sync
```

## 快速开始

### 方式一：基于规则的控制

#### 步骤 1: 启动 Chrome 恐龙游戏
1. 打开 Chrome 浏览器
2. 访问 `chrome://dino/` 或断开网络后访问任意网页
3. 按空格键开始游戏，让恐龙跑起来

#### 步骤 2: 运行规则控制程序

```bash
uv run python play_rule_based.py
```

首次运行会提示：
1. **选择显示器**: 输入显示器编号 (通常是 0)
2. **选择游戏区域**: 用鼠标框选恐龙游戏区域，按 **ENTER** 确认

> 配置会自动保存到 `config.json`，下次运行无需重新选择

#### 步骤 3: 观察运行效果
- 程序会显示游戏画面窗口，带有检测框和标签
- 底部显示当前按键状态指示器
- 按 **Q** 键退出程序

### 方式二：强化学习控制

#### 训练模型

```bash
uv run python train_rl.py
```

训练参数（可通过命令行参数修改）：
- `--timesteps`: 总训练步数（默认 1000000）
- `--episodes`: 最大训练回合数（默认 0，表示无限制）
- `--lr`: 学习率（默认 3e-4）
- `--save-freq`: 检查点保存频率（默认 10000）
- `--save-path`: 模型保存路径（默认 weights/rl）
- `--log-dir`: TensorBoard 日志目录（默认 logs）
- `--no-render`: 禁用训练时的可视化窗口（默认启用）
- `--only-up`: 仅使用跳跃动作（不使用下蹲，动作空间变为2个）

训练过程中：
- 模型和日志目录自动添加时间戳（格式：`YYMMDD_HHMMSS`）
- 如果指定了 `--episodes`，训练到指定回合数后自动停止
- 按 **Q** 键可随时停止训练并保存当前模型
- 模型自动保存到 `weights/rl/<timestamp>/checkpoints/`
- 最优模型保存到 `weights/rl/<timestamp>/best/model.zip`
- 最终模型保存到 `weights/rl/<timestamp>/final_model.zip`

#### 使用训练好的模型玩游戏

```bash
# 使用最新的训练模型
uv run python play_rl.py --latest

# 指定具体模型路径
uv run python play_rl.py --weights weights/rl/250418_143052/best/model.zip
```

#### 查看训练曲线

```bash
tensorboard --logdir logs/
```

## 项目结构

```
chrome-dino-bot/
├── src/
│   ├── core/                    # 核心共享模块
│   │   ├── detector.py          # YOLO检测器封装
│   │   ├── screen.py            # 屏幕截图功能
│   │   └── keyboard.py          # 键盘控制功能
│   │
│   ├── rule_based/              # 基于规则的控制方法
│   │   ├── controller.py        # 规则控制器
│   │   └── play.py              # 游戏运行入口
│   │
│   ├── rl/                      # 强化学习模块
│   │   ├── env.py               # Gymnasium环境封装
│   │   ├── callbacks.py         # SB3回调函数
│   │   ├── train.py             # 训练脚本
│   │   └── play.py              # 使用训练模型玩游戏
│   │
│   └── utils/                   # 工具函数
│       └── visualization.py     # 可视化工具
│
├── weights/                     # 模型权重目录
│   ├── yolo26n_dino_260418.pt   # YOLO检测模型
│   └── rl/                      # RL模型权重
│       └── <timestamp>/         # 训练时间戳目录 (YYMMDD_HHMMSS)
│           ├── best/            # 最优权重
│           ├── checkpoints/     # 检查点
│           └── final_model.zip  # 最终模型
│
├── logs/                        # TensorBoard日志
│   └── <timestamp>/             # 训练时间戳目录 (YYMMDD_HHMMSS)
├── assets/                      # UI 图标资源
├── dataset/                     # 训练数据集目录
├── runs/                        # YOLO训练结果
├── play_rule_based.py           # 规则方法入口脚本
├── train_rl.py                  # RL训练入口脚本
├── play_rl.py                   # RL游戏入口脚本
├── config.json                  # ROI 配置文件
├── pyproject.toml               # 项目配置
└── AGENTS.md                    # 本文档
```

## 核心模块说明

### 1. src/core/detector.py - YOLO检测器

封装 YOLO26 模型，提供统一的检测接口：

```python
from src.core import DinoDetector

detector = DinoDetector(model_path="weights/yolo26n_dino_260418.pt")
result = detector.detect(image)

# 结果包含：
# - result.dino: 恐龙检测结果
# - result.obstacles: 障碍物列表（仙人掌、飞鸟）
# - result.has_restart: 是否检测到重新开始标签
```

### 2. src/core/screen.py - 屏幕截图

提供屏幕截图和ROI选择功能：

```python
from src.core import capture_screenshot, select_screen_and_roi

# 捕获截图
image = capture_screenshot()

# 重新选择游戏区域
select_screen_and_roi()
```

### 3. src/core/keyboard.py - 键盘控制

封装键盘操作，支持跳跃和下蹲：

```python
from src.core import KeyboardController

keyboard = KeyboardController()
keyboard.press_jump()    # 跳跃（持续0.5秒）
keyboard.press_duck()    # 下蹲（支持长按）
keyboard.press_enter()   # 按回车键
keyboard.update()        # 更新按键状态
```

### 4. src/rl/env.py - Gymnasium环境

实现强化学习环境接口：

**状态空间（12维向量）**：
| 特征 | 描述 | 归一化方式 | 缺失值 |
|------|------|-----------|--------|
| dino_y | 恐龙Y坐标 | / 图像高度 | 0.5 |
| obstacle_1_label | 最近障碍物标签 | 仙人掌=0, 飞鸟=1 | -1 |
| obstacle_1_x/y/w/h | 障碍物位置和尺寸 | / 图像尺寸 | -1 |
| obstacle_2_label | 第二近障碍物标签 | 同上 | -1 |
| obstacle_2_x/y/w/h | 障碍物位置和尺寸 | / 图像尺寸 | -1 |
| speed | 障碍物移动速度 | / 1500 | 0 |

**动作空间（3个离散动作）**：
| 动作ID | 动作 | 描述 |
|--------|------|------|
| 0 | NOOP | 不按键 |
| 1 | UP | 跳跃 |
| 2 | DOWN | 下蹲 |

**奖励函数**：
- 存活奖励：+0.01（每帧）
- 通过障碍物：+1.0（每个障碍物）
- 游戏结束：-2.0

### 5. src/rule_based/controller.py - 规则控制器

基于规则的决策逻辑：

| 检测对象 | 条件 | 动作 |
|---------|------|------|
| 仙人掌 | Y中心点在 100-190，X距离在触发范围内 | 跳跃 |
| 飞鸟 (低位) | Y中心点 > 150，X距离在触发范围内 | 跳跃 |
| 飞鸟 (高位) | Y中心点 <= 150，X距离在触发范围内 | 下蹲 |

## YOLO26 模型

使用本地训练的 YOLO26 模型：
- **模型路径**: `weights/yolo26n_dino_260418.pt`
- **检测类别**: `dino` (恐龙), `cactus` (仙人掌), `bird` (飞鸟), `restart` (重新开始)

## 配置文件

### config.json (运行时生成)

```json
{
    "top": 100,
    "left": 200,
    "width": 600,
    "height": 150,
    "mon": 1
}
```

## 依赖说明

| 包名 | 用途 |
|-----|------|
| ultralytics | YOLO26 模型推理 |
| stable-baselines3 | PPO 强化学习算法 |
| gymnasium | 强化学习环境接口 |
| pynput | 键盘事件模拟 |
| supervision | 目标检测可视化标注 |
| opencv-python | 图像处理 |
| mss | 屏幕截图捕获 |
| numpy | 数值计算 |
| keyboard | 按键检测（用于训练中断） |

## 扩展开发建议

1. **改进奖励函数**: 调整奖励参数或添加新的奖励项
2. **自定义网络架构**: 修改 `src/rl/train.py` 中的 `policy_kwargs`
3. **多游戏支持**: 扩展 `src/rl/env.py` 以支持其他游戏
4. **数据采集**: 使用 `src/core/screen.py` 采集更多训练数据

## 常见问题

### Q: 如何重新选择游戏区域？
删除 `config.json` 文件后重新运行程序。

### Q: 检测不准确怎么办？
- 确保游戏区域选择正确
- 调整 `src/core/detector.py` 中的置信度阈值（默认 0.6）
- 采集更多数据训练自定义 YOLO 模型

### Q: 训练不收敛怎么办？
- 调整学习率（尝试 1e-4 到 1e-3）
- 增加训练步数
- 检查奖励函数设计是否合理

### Q: GPU 加速不生效？
确保安装了 CUDA 和 cuDNN，PyTorch 会自动使用 GPU。
