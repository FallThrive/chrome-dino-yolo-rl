# Chrome Dino YOLO RL

简体中文 | [English](README.md)

一个使用 YOLO26 目标检测模型和强化学习（PPO 算法）的 Chrome 恐龙游戏机器人项目。项目支持两种控制方式：

1. **基于规则的控制**：使用预设规则进行游戏决策
2. **强化学习控制**：使用 PPO（Proximal Policy Optimization）算法训练智能体

## 技术栈

- **Python 3.12+**
- **YOLO26** - 目标检测模型（通过 Ultralytics）
- **Stable Baselines3** - 强化学习框架（PPO 算法）
- **Gymnasium** - 强化学习环境接口
- **OpenCV** - 图像处理
- **mss** - 屏幕截图捕获
- **keyboard** - 键盘控制
- **supervision** - 目标检测可视化

## 环境配置

本项目使用 `uv` 作为包管理器。

### 克隆仓库

```bash
git clone https://github.com/FallThrive/chrome-dino-yolo-rl.git
cd chrome-dino-yolo-rl
```

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
uv run python src/rule_based/play.py
```

首次运行会提示：

1. **选择显示器**：输入显示器编号（通常是 0）
2. **选择游戏区域**：用鼠标框选恐龙游戏区域，按 **ENTER** 确认

> 配置会自动保存到 `cfg/roi.json`，下次运行无需重新选择

#### 步骤 3: 观察运行效果

- 程序会显示游戏画面窗口，带有检测框和标签
- 底部显示当前按键状态指示器
- 按 **Q** 键退出程序

### 方式二：强化学习控制

#### 训练模型

```bash
uv run python src/rl/train.py
```

训练参数（可通过命令行参数修改）：

- `--timesteps`: 总训练步数（默认 1000000）
- `--episodes`: 最大训练回合数（默认 0，表示无限制）
- `--lr`: 学习率（默认 3e-4）
- `--save-freq`: 检查点保存频率（默认 10000）
- `--save-path`: 模型保存路径（默认 weights/rl）
- `--log-dir`: TensorBoard 日志目录（默认 logs）
- `--no-render`: 禁用训练时的可视化窗口（默认启用）
- `--only-up`: 仅使用跳跃动作（不使用下蹲，动作空间变为 2 个）

训练过程中：

- 模型和日志目录自动添加时间戳（格式：`YYMMDD_HHMMSS`）
- 如果指定了 `--episodes`，训练到指定回合数后自动停止
- 按 **Q** 键可随时停止训练并保存当前模型
- 模型自动保存到 `weights/rl/<timestamp>/checkpoints/`
- 最优模型保存到 `weights/rl/<timestamp>/best/model.zip`
- 最终模型保存到 `weights/rl/<timestamp>/final_model.zip`

#### 使用训练好的模型进行游戏

由于不同ROI下的模型训练结果不同，因此这里不提供默认训练的PPO模型。

```bash
# 使用最新的训练模型
uv run python src/rl/play.py --latest

# 指定具体模型路径
uv run python src/rl/play.py --weights weights/rl/250418_143052/best/model.zip
```

#### 查看训练曲线

```bash
tensorboard --logdir logs/
```

### 数据采集

用于采集游戏截图以训练 YOLO 模型：

```bash
uv run python src/core/take_screenshots.py
```

运行后会：

1. 提示选择显示器
2. 选择游戏区域 ROI
3. 自动循环截图并保存到 `dataset/` 目录

## 项目结构

```
chrome-dino-yolo-rl/
├── src/
│   ├── core/                    # 核心共享模块
│   │   ├── __init__.py
│   │   ├── detector.py          # YOLO 检测器封装
│   │   ├── keyboard.py          # 键盘控制功能
│   │   ├── screen.py            # 屏幕截图功能
│   │   └── take_screenshots.py  # 数据采集脚本
│   │
│   ├── rule_based/              # 基于规则的控制方法
│   │   ├── __init__.py
│   │   ├── controller.py        # 规则控制器
│   │   └── play.py              # 游戏运行入口
│   │
│   ├── rl/                      # 强化学习模块
│   │   ├── __init__.py
│   │   ├── callbacks.py         # SB3 回调函数
│   │   ├── env.py               # Gymnasium 环境封装
│   │   ├── play.py              # 使用训练模型玩游戏
│   │   └── train.py             # 训练脚本
│   │
│   ├── utils/                   # 工具函数
│   │   ├── __init__.py
│   │   └── visualization.py     # 可视化工具
│   │
│   ├── yolo/                    # YOLO 训练脚本
│   │   ├── train_yolo26n.py     # 标准训练脚本
│   │   └── train_yolo26n_simple.py  # 简化版训练脚本
│   │
│   └── __init__.py
│
├── weights/                     # 模型权重目录
│   ├── yolo26n_dino.pt          # YOLO 检测模型
│   ├── yolo26n_dino_simple.pt   # YOLO 简化版检测模型
│   └── rl/                      # RL 模型权重
│       └── <timestamp>/         # 训练时间戳目录 (YYMMDD_HHMMSS)
│           ├── best/            # 最优权重
│           ├── checkpoints/     # 检查点
│           └── final_model.zip  # 最终模型
│
├── logs/                        # TensorBoard 日志
│   └── <timestamp>/             # 训练时间戳目录 (YYMMDD_HHMMSS)
├── cfg/                         # 配置文件
│   ├── roi.json                 # ROI 配置
│   └── yolo26n_simple.yaml      # YOLO 模型配置
├── dataset/                     # 训练数据集目录
├── runs/                        # YOLO 训练结果
├── assets/                      # UI 图标资源
├── pyproject.toml               # 项目配置
├── requirements.txt             # 依赖列表
└── README_CN.md                 # 本文档（中文）
```

## YOLO26 模型

使用本地训练的 YOLO26 模型：

- **原始网络模型路径**: `weights/yolo26n_dino.pt`
- **简化网络模型路径**: `weights/yolo26n_dino_simple.pt`
- **检测类别**: `dino` (恐龙), `cactus` (仙人掌), `bird` (飞鸟), `restart` (重新开始)

## 配置文件

### roi.json (运行时生成)

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

| 包名                    | 用途          |
| --------------------- | ----------- |
| ultralytics           | YOLO26 模型推理 |
| stable-baselines3     | PPO 强化学习算法  |
| gymnasium             | 强化学习环境接口    |
| keyboard              | 键盘事件模拟      |
| supervision           | 目标检测可视化标注   |
| opencv-contrib-python | 图像处理        |
| mss                   | 屏幕截图捕获      |
| numpy                 | 数值计算        |
| torch                 | 深度学习框架      |
| torchvision           | PyTorch 视觉库 |

## 常见问题

### Q: 如何重新选择游戏区域？

删除 `cfg/roi.json` 文件后重新运行程序。

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

## 扩展开发建议

1. **改进奖励函数**: 调整奖励参数或添加新的奖励项
2. **自定义网络架构**: 修改 `src/rl/train.py` 中的 `policy_kwargs`
3. **多游戏支持**: 扩展 `src/rl/env.py` 以支持其他游戏
4. **数据采集**: 使用 `src/core/take_screenshots.py` 采集更多训练数据

---

## 致谢 Acknowledgements

本项目基于 [Erol444/chrome-dino-bot](https://github.com/Erol444/chrome-dino-bot) 项目重构而来。

This project is refactored based on [Erol444/chrome-dino-bot](https://github.com/Erol444/chrome-dino-bot).
