import torch
from ultralytics import YOLO


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if device == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

model = YOLO('weights/yolo26n.pt')

model.train(
    data='data/dino_260417/dino_260417.yaml', # 数据集配置文件
    epochs=300, # 训练轮次
    imgsz=640, # 图像大小
    batch=40, # 批次大小：-1表示自动调整
    device=device, # 设备
    workers=16, # 数据加载线程数
    # project='train', # 训练结果保存目录
    name='dino_260417', # 实验名称
    exist_ok=True, # 是否覆盖已有实验
    amp=True, # 启用自动混合精度
    patience=50 # 早停机制的耐心值：0表示禁用
)
