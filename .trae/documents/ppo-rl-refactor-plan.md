# PPO强化学习重构计划

## 项目概述

将Chrome恐龙游戏机器人从基于规则的控制方法重构为使用PPO（Proximal Policy Optimization）强化学习方法。

## 一、项目结构重组

### 新的目录结构

```
chrome-dino-bot/
├── src/
│   ├── core/                    # 核心共享模块
│   │   ├── __init__.py
│   │   ├── detector.py          # YOLO检测器封装
│   │   ├── screen.py            # 屏幕截图相关
│   │   └── keyboard.py          # 键盘控制
│   │
│   ├── rule_based/              # 原有的基于规则的控制方法
│   │   ├── __init__.py
│   │   ├── controller.py        # 规则控制器
│   │   └── play.py              # 原有游戏运行入口
│   │
│   ├── rl/                      # 强化学习模块
│   │   ├── __init__.py
│   │   ├── env.py               # Gymnasium环境封装
│   │   ├── callbacks.py         # SB3回调函数
│   │   ├── train.py             # 训练脚本
│   │   └── play.py              # 使用训练模型玩游戏
│   │
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       └── visualization.py     # 可视化工具
│
├── weights/                     # 模型权重目录
│   └── rl/                      # RL模型权重
│       ├── best/                # 最优权重
│       └── checkpoints/         # 检查点
│
├── logs/                        # TensorBoard日志
├── assets/                      # UI资源（保持不变）
├── dataset/                     # 数据集（保持不变）
├── runs/                        # YOLO训练结果（保持不变）
├── pyproject.toml               # 更新依赖
├── config.json                  # ROI配置（保持不变）
└── AGENTS.md                    # 更新项目文档
```

## 二、状态空间设计

### 技术选型：使用 Stable Baselines3

**选择原因**：
- 成熟稳定的PPO实现，经过充分测试和优化
- 原生支持TensorBoard集成
- 支持自定义Gym环境
- 活跃的社区和完善的文档
- 易于保存/加载模型和检查点

### 状态向量（归一化处理）

| 特征 | 描述 | 归一化方式 | 缺失时默认值 |
|------|------|-----------|-------------|
| dino_y | 恐龙纵坐标中心点 | / 图像高度 | 0.5 |
| obstacle_1_label | 最近障碍物标签 | 仙人掌=0, 飞鸟=1, 无=-1 | -1 |
| obstacle_1_x | 最近障碍物x坐标 | / 图像宽度 | -1 |
| obstacle_1_y | 最近障碍物y中心点 | / 图像高度 | -1 |
| obstacle_1_w | 最近障碍物宽度 | / 图像宽度 | -1 |
| obstacle_1_h | 最近障碍物高度 | / 图像高度 | -1 |
| obstacle_2_label | 第二近障碍物标签 | 同上 | -1 |
| obstacle_2_x | 第二近障碍物x坐标 | / 图像宽度 | -1 |
| obstacle_2_y | 第二近障碍物y中心点 | / 图像高度 | -1 |
| obstacle_2_w | 第二近障碍物宽度 | / 图像宽度 | -1 |
| obstacle_2_h | 第二近障碍物高度 | / 图像高度 | -1 |
| speed | 当前障碍物移动速度 | / 1500 (最大速度) | 0 |

**状态维度**: 12维向量

### 障碍物缺失处理策略

```python
def build_state(dino_y, obstacles, speed, img_w, img_h):
    """
    构建状态向量，处理障碍物缺失情况
    
    obstacles: 按x坐标排序的障碍物列表（只包含x>40的）
    缺失值统一使用 -1 表示
    """
    state = np.full(12, -1.0)  # 初始化为 -1
    
    # 恐龙Y坐标
    state[0] = dino_y / img_h if dino_y else 0.5
    
    # 第一个障碍物
    if len(obstacles) >= 1:
        obs = obstacles[0]
        state[1] = 0 if obs.label == 'cactus' else 1  # 标签
        state[2] = obs.x / img_w
        state[3] = obs.y_center / img_h
        state[4] = obs.width / img_w
        state[5] = obs.height / img_h
    # 否则保持 -1
    
    # 第二个障碍物
    if len(obstacles) >= 2:
        obs = obstacles[1]
        state[6] = 0 if obs.label == 'cactus' else 1
        state[7] = obs.x / img_w
        state[8] = obs.y_center / img_h
        state[9] = obs.width / img_w
        state[10] = obs.height / img_h
    # 否则保持 -1
    
    # 速度（速度为0是有效的，不使用-1）
    state[11] = speed / 1500.0
    
    return state.astype(np.float32)
```

## 三、动作空间设计

### 离散动作空间（3个动作）

| 动作ID | 动作 | 描述 |
|--------|------|------|
| 0 | NOOP | 不按键 |
| 1 | UP | 按上键（跳跃） |
| 2 | DOWN | 按下键（下蹲） |

## 四、奖励函数设计

### 完善的奖励函数

```python
# 奖励参数
SURVIVAL_REWARD = 0.01          # 存活奖励（每帧）
OBSTACLE_PASS_REWARD = 1.0      # 通过障碍物奖励
GAME_OVER_PENALTY = -2.0        # 游戏结束惩罚（通过障碍物奖励的双倍）
```

### 奖励计算逻辑

1. **存活奖励（稠密）**：
   - 每帧给予小正奖励 `+0.01`
   - 鼓励智能体存活更长时间

2. **通过障碍物奖励（稀疏）**：
   - 当障碍物从右侧移动到恐龙左侧（x <= 40）时触发
   - 每个障碍物只奖励一次（使用集合记录已通过的障碍物）
   - 奖励值：`+1.0`

3. **游戏结束惩罚**：
   - 检测到 `restart` 标签时触发
   - 惩罚值：`-2.0`（通过障碍物奖励的双倍）
   - 结束当前episode

### 奖励函数伪代码

```python
def calculate_reward(self, detections, passed_obstacles):
    reward = SURVIVAL_REWARD  # 存活奖励
    
    # 检查通过的障碍物
    for obstacle in obstacles_right_of_dino:
        if obstacle.x <= 40 and obstacle.id not in passed_obstacles:
            reward += OBSTACLE_PASS_REWARD
            passed_obstacles.add(obstacle.id)
    
    # 检查游戏结束
    if has_restart_label(detections):
        reward += GAME_OVER_PENALTY
        done = True
    
    return reward, done, passed_obstacles
```

## 五、PPO算法实现（使用Stable Baselines3）

### 使用Stable Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# 创建环境
env = DinoGameEnv()

# 创建PPO模型
model = PPO(
    "MlpPolicy",           # 多层感知机策略
    env,
    learning_rate=3e-4,
    n_steps=2048,          # 每次更新收集的步数
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    tensorboard_log="./logs/",
    verbose=1,
)

# 自定义回调：处理游戏重置
class DinoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self):
        # 当环境返回done=True时，按回车键重新开始
        if self.locals.get('dones', [False])[0]:
            press_enter_to_restart()
        return True

# 训练
model.learn(
    total_timesteps=1000000,
    callback=[
        CheckpointCallback(save_freq=10000, save_path="./weights/rl/checkpoints/"),
        DinoCallback(),
    ],
)
```

### 网络架构配置（可选自定义）

```python
from stable_baselines3.common.vec_env import DummyVecEnv

policy_kwargs = dict(
    net_arch=[dict(pi=[128, 64], vf=[128, 64])]  # Actor和Critic网络架构
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)
```

### 超参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 3e-4 | 学习率 |
| n_steps | 2048 | 每次更新收集的步数 |
| batch_size | 64 | 批大小 |
| n_epochs | 10 | 每次更新的epoch数 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE参数 |
| clip_range | 0.2 | PPO裁剪参数 |
| ent_coef | 0.01 | 熵系数（鼓励探索） |
| vf_coef | 0.5 | 价值损失系数 |

## 六、训练流程

### Episode流程（Gymnasium环境）

```
1. 环境初始化（reset）：
   - 检测游戏状态
   - 如果检测到 restart 标签，按回车键重新开始
   - 返回初始状态

2. 每一步（step）：
   a. 接收动作（0=不按键, 1=跳跃, 2=下蹲）
   b. 执行按键操作
   c. 等待一帧
   d. 获取新截图和检测结果
   e. 计算奖励和是否结束
   f. 返回 (next_state, reward, terminated, truncated, info)

3. 游戏结束处理（在回调函数中）：
   a. 当 terminated=True 时
   b. 按回车键重新开始游戏
   c. 环境自动重置
```

### SB3回调函数设计

```python
class DinoTrainingCallback(BaseCallback):
    """
    自定义回调函数：
    1. 处理游戏重置（按回车键）
    2. 保存最优模型
    3. 处理Q键退出
    """
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -float('inf')
    
    def _on_step(self):
        # 检查Q键退出
        if keyboard.is_pressed('q'):
            self.model.save(f"{self.save_path}/final_model.zip")
            return False  # 停止训练
        
        # 当episode结束时按回车重置游戏
        for done in self.locals.get('dones', []):
            if done:
                press_enter_key()
                break
        
        return True
    
    def _on_rollout_end(self):
        # 检查是否为最优模型
        mean_reward = np.mean(self.model.ep_info_buffer[-100:])
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(f"{self.save_path}/best/model.zip")
```

### 检查点保存策略

- 自动保存：每 `save_freq` 步保存到 `weights/rl/checkpoints/`
- 最优模型：当平均奖励超过历史最优时保存到 `weights/rl/best/model.zip`
- 手动退出：按Q键时保存当前模型到 `weights/rl/final_model.zip`
- TensorBoard日志自动保存到 `logs/`

## 七、依赖更新

### 新增依赖

```toml
[project.dependencies]
# 现有依赖保持不变
# 新增：
stable-baselines3 = ">=2.0.0"
gymnasium = ">=0.29.0"  # SB3使用gymnasium作为环境接口
```

## 八、实施步骤

### 步骤1：创建目录结构
- 创建 `src/` 目录及子目录
- 创建 `weights/rl/` 目录
- 创建 `logs/` 目录

### 步骤2：实现核心模块
- `src/core/detector.py`：封装YOLO检测器
- `src/core/screen.py`：屏幕截图功能
- `src/core/keyboard.py`：键盘控制功能

### 步骤3：迁移原有代码
- 将 `controller.py` 移动到 `src/rule_based/`
- 将 `play_game.py` 重命名为 `play.py` 并移动到 `src/rule_based/`
- 将 `utils.py` 移动到 `src/utils/visualization.py`
- 将 `take_screenshots.py` 移动到 `src/core/screen.py`

### 步骤4：实现强化学习模块
- `src/rl/env.py`：实现Gymnasium风格的游戏环境
- `src/rl/callbacks.py`：实现SB3回调函数（游戏重置、最优模型保存）
- `src/rl/train.py`：实现训练脚本（使用SB3 PPO）
- `src/rl/play.py`：实现使用训练模型玩游戏的脚本

### 步骤5：创建入口脚本
- `play_rule_based.py`：运行基于规则的方法
- `train_rl.py`：训练强化学习模型
- `play_rl.py`：使用训练好的模型玩游戏

### 步骤6：更新配置
- 更新 `pyproject.toml` 添加新依赖
- 更新 `AGENTS.md` 文档

### 步骤7：清理旧文件
- 删除根目录下的旧Python文件（已迁移）

## 九、关键实现细节

### 障碍物ID追踪

为了正确判断障碍物是否通过，需要为每个障碍物分配唯一ID：

```python
class ObstacleTracker:
    def __init__(self):
        self.next_id = 0
        self.active_obstacles = {}  # id -> last_position
    
    def update(self, detections):
        # 基于位置匹配障碍物，分配ID
        pass
    
    def get_passed_obstacles(self, threshold_x=40):
        # 返回已通过恐龙的障碍物ID列表
        pass
```

### 状态归一化

```python
def normalize_state(state, image_width, image_height):
    normalized = np.zeros(12)
    normalized[0] = state['dino_y'] / image_height
    # ... 其他特征归一化
    return normalized
```

### 按键逻辑说明

```python
KEY_PRESS_DURATION = {
    'up': 0.5,     # 跳跃从起跳到下降需要0.5秒
    'down': 0.1,   # 下蹲时间极短，但支持长按
}

LONG_PRESS_SUPPORT = {
    'up': True,    # 允许长按上键（跳跃过程中继续按不会打断）
    'down': True,  # 允许长按下键（下蹲需要长按）
}
```

**按键行为**：
- **跳跃（上键）**：按下后持续0.5秒完成跳跃动作，期间继续按上键或下键不会打断
- **下蹲（下键）**：时间极短但支持长按，长按时持续下蹲状态
- **不按键**：释放所有按键，恢复正常状态

## 十、文件清单

### 新建文件

| 文件路径 | 描述 |
|---------|------|
| `src/core/__init__.py` | 核心模块初始化 |
| `src/core/detector.py` | YOLO检测器封装 |
| `src/core/screen.py` | 屏幕截图功能 |
| `src/core/keyboard.py` | 键盘控制功能 |
| `src/rule_based/__init__.py` | 规则模块初始化 |
| `src/rule_based/controller.py` | 规则控制器 |
| `src/rule_based/play.py` | 规则方法游戏入口 |
| `src/rl/__init__.py` | RL模块初始化 |
| `src/rl/env.py` | Gymnasium环境封装 |
| `src/rl/callbacks.py` | SB3回调函数（游戏重置、最优模型保存） |
| `src/rl/train.py` | 训练脚本 |
| `src/rl/play.py` | RL游戏入口 |
| `src/utils/__init__.py` | 工具模块初始化 |
| `src/utils/visualization.py` | 可视化工具 |
| `play_rule_based.py` | 规则方法入口脚本 |
| `train_rl.py` | RL训练入口脚本 |
| `play_rl.py` | RL游戏入口脚本 |

### 移动/重命名文件

| 原路径 | 新路径 |
|-------|-------|
| `controller.py` | `src/rule_based/controller.py` |
| `play_game.py` | `src/rule_based/play.py` |
| `utils.py` | `src/utils/visualization.py` |
| `take_screenshots.py` | 合并到 `src/core/screen.py` |
| `test_model.py` | `src/core/test_detector.py` |

### 删除文件

| 文件路径 | 原因 |
|---------|------|
| `controller.py` | 已移动 |
| `play_game.py` | 已移动 |
| `utils.py` | 已移动 |
| `take_screenshots.py` | 已合并 |
| `test_model.py` | 已移动 |

## 十一、使用说明

### 安装依赖

```bash
uv sync
```

### 运行基于规则的方法

```bash
uv run python play_rule_based.py
```

### 训练强化学习模型

```bash
uv run python train_rl.py
```

训练参数（可在 `train_rl.py` 中修改）：
- `total_timesteps`: 总训练步数（默认100万）
- `learning_rate`: 学习率（默认3e-4）
- `save_freq`: 检查点保存频率（默认10000步）

### 使用训练好的模型玩游戏

```bash
uv run python play_rl.py --weights weights/rl/best/model.zip
```

### 查看训练曲线

```bash
tensorboard --logdir logs/
```

TensorBoard将显示：
- 奖励曲线（ep_rew_mean）
- 损失曲线（policy_loss, value_loss）
- 熵曲线（entropy_loss）
