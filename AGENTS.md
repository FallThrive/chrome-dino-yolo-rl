# Chrome Dino Bot - AI Agent 项目指南

## 项目简介

这是一个使用 YOLOv8 目标检测模型自动玩 Chrome 恐龙游戏的 AI 机器人项目。项目通过实时屏幕捕获、目标检测和键盘控制实现游戏自动化。

## 技术栈

- **Python 3.12+**
- **YOLOv8** - 目标检测模型 (通过 Roboflow inference)
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

### 首次运行流程

#### 步骤 1: 启动 Chrome 恐龙游戏
1. 打开 Chrome 浏览器
2. 访问 `chrome://dino/` 或断开网络后访问任意网页
3. 按空格键开始游戏，让恐龙跑起来

#### 步骤 2: 运行主程序并配置区域

```bash
uv run python play_game.py
```

首次运行会提示：
1. **选择显示器**: 输入显示器编号 (通常是 0)
2. **选择游戏区域**: 用鼠标框选恐龙游戏区域，按 **ENTER** 确认

> 配置会自动保存到 `config.json`，下次运行无需重新选择

#### 步骤 3: 观察运行效果
- 程序会显示游戏画面窗口，带有检测框和标签
- 底部显示当前按键状态指示器
- 按 **Q** 键退出程序

### 测试模型 (可选)

如果想先测试模型检测效果，不执行键盘操作：

```bash
uv run python test_model.py
```

### 数据采集 (可选)

如果需要采集数据集训练自定义模型：

```bash
uv run python take_screenshots.py
```

截图会自动保存到 `dataset/` 目录，每 0.5 秒一张。

## 项目结构

```
chrome-dino-bot/
├── play_game.py        # 主程序入口
├── controller.py       # 游戏决策逻辑
├── take_screenshots.py # 数据集采集工具
├── test_model.py       # 模型推理测试
├── utils.py            # 工具函数
├── assets/             # UI 图标资源
├── dataset/            # 训练数据集目录
├── config.json         # ROI 配置文件 (运行时生成)
├── pyproject.toml      # 项目配置
└── uv.lock             # 依赖锁定文件
```

## 核心模块说明

### 1. play_game.py - 主程序

主循环逻辑：
- 加载 YOLOv8 模型 (`dino-game-rcopt/14`)
- 持续捕获游戏截图
- 运行目标检测推理
- 根据检测结果执行键盘操作
- 显示带标注的游戏画面

关键代码流程：
```python
model = get_model(model_id="dino-game-rcopt/14")
while True:
    image = capture_screenshot()
    results = model.infer(image)
    detections = sv.Detections.from_inference(results[0])
    action = get_action(detections)
    # 执行动作: "up" (跳跃) 或 "down" (下蹲)
```

### 2. controller.py - 决策控制器

基于规则的决策逻辑：

| 检测对象 | 条件 | 动作 |
|---------|------|------|
| 仙人掌 | Y中心点在 110-144，X左边界在 130-170 | 跳跃 (空格) |
| 飞鸟 (低位) | X在 100-200，Y中心点 > 121 | 跳跃 (空格) |
| 飞鸟 (高位) | X在 100-200，Y中心点 <= 121 | 下蹲 (下箭头) |

可扩展方向：
- 使用强化学习替代规则系统
- 实现进化算法优化决策

### 3. take_screenshots.py - 数据采集

功能：
- 首次运行时选择显示器和游戏区域 (ROI)
- 配置保存到 `config.json`
- 持续截图保存到 `dataset/` 目录

### 4. utils.py - UI 工具

- 加载键盘图标资源
- 绘制当前按键状态指示器
- 图像叠加处理

## YOLOv8 模型

使用 Roboflow 托管的预训练模型：
- **Model ID**: `dino-game-rcopt/14`
- **检测类别**: `cactus` (仙人掌), `bird` (飞鸟)
- **模型来源**: https://universe.roboflow.com/erol4/dino-game-rcopt

模型会在首次运行时自动下载到本地。

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
| inference-gpu | YOLOv8 模型推理 (GPU 加速) |
| pynput | 键盘事件模拟 |
| supervision | 目标检测可视化标注 |
| opencv-python | 图像处理 (通过依赖传递) |
| mss | 屏幕截图捕获 |
| numpy | 数值计算 |

> 注意：如果没有 NVIDIA GPU，可将 `inference-gpu` 替换为 `inference`。

## 扩展开发建议

1. **改进决策系统**: 将 `controller.py` 中的规则替换为神经网络或强化学习
2. **自定义模型**: 使用采集的数据集训练专属 YOLOv8 模型
3. **多游戏支持**: 扩展架构以支持其他游戏的自动化
4. **性能优化**: 使用多线程分离截图和推理过程

## 常见问题

### Q: 如何重新选择游戏区域？
删除 `config.json` 文件后重新运行程序。

### Q: 检测不准确怎么办？
- 确保游戏区域选择正确
- 调整 `play_game.py` 中的置信度阈值 (默认 0.6)
- 使用 `take_screenshots.py` 采集更多数据训练自定义模型

### Q: GPU 加速不生效？
确保安装了 CUDA 和 cuDNN，并使用 `inference-gpu` 包。
