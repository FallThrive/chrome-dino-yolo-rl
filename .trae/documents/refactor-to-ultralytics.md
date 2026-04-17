# 重构计划：从 Roboflow 迁移到 Ultralytics YOLO

## 背景
将模型加载方式从 Roboflow `inference` 库迁移到 `ultralytics` 库，使用本地模型文件 `weights/yolo26n_dino_260418.pt`。

## 检测标签
- `bird` - 飞鸟
- `cactus` - 仙人掌
- `dino` - 恐龙（新增）
- `restart` - 重启按钮（新增）

---

## 实施步骤

### 步骤 1: 修改依赖配置文件

#### 1.1 修改 `requirements.txt`
- 移除 `inference-gpu` 行
- 添加 `ultralytics`

#### 1.2 修改 `pyproject.toml`
- 移除 `inference-gpu>=0.26.1` 依赖
- 添加 `ultralytics` 依赖

### 步骤 2: 修改 `play_game.py`

#### 2.1 修改导入语句
```python
# 移除
from inference import get_model

# 添加
from ultralytics import YOLO
```

#### 2.2 修改模型加载
```python
# 移除
model = get_model(model_id="dino-game-rcopt/14")

# 改为
model = YOLO("weights/yolo26n_dino_260418.pt")
```

#### 2.3 修改推理调用
```python
# 移除
results = model.infer(image)
detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

# 改为
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)
```

### 步骤 3: 修改 `test_model.py`

#### 3.1 修改导入语句
同上，移除 `inference` 导入，添加 `ultralytics` 导入

#### 3.2 修改模型加载和推理
同上，使用 `YOLO` 类加载本地模型

#### 3.3 修改检测结果处理
```python
# 移除
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)

# 改为
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)
```

### 步骤 4: 修改 `controller.py`

#### 4.1 更新恐龙位置常量
```python
# 修改
DINO_X_POSITION = 100

# 改为
DINO_X_POSITION = 40
```

#### 4.2 标签过滤逻辑
现有代码已正确处理 `cactus` 和 `bird` 标签，新增的 `dino` 和 `restart` 标签不影响现有逻辑（因为只处理障碍物）。

### 步骤 5: 环境重建

#### 5.1 删除现有虚拟环境
```bash
# 删除 .venv 目录
rmdir /s /q .venv
```

#### 5.2 重新安装依赖
```bash
uv sync
```

---

## 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `requirements.txt` | 编辑 | 移除 inference-gpu，添加 ultralytics |
| `pyproject.toml` | 编辑 | 移除 inference-gpu，添加 ultralytics |
| `play_game.py` | 编辑 | 修改模型加载和推理方式 |
| `test_model.py` | 编辑 | 修改模型加载和推理方式 |
| `controller.py` | 编辑 | 修改 DINO_X_POSITION 为 40 |

---

## 注意事项

1. **API 差异**：
   - Roboflow `inference`: `model.infer(image)` 返回列表
   - Ultralytics: `model(image)` 返回 `Results` 对象列表

2. **supervision 兼容性**：
   - `sv.Detections.from_ultralytics()` 方法需要 supervision >= 0.22.0
   - 当前版本已满足要求

3. **模型路径**：
   - 使用相对路径 `weights/yolo26n_dino_260418.pt`
   - 确保模型文件存在

4. **依赖冲突**：
   - `inference-gpu` 与 `ultralytics` 存在冲突
   - 必须完全移除 `inference-gpu` 后重装环境
