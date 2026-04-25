import torch
from ultralytics import YOLO


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    model = YOLO('cfg/yolo26n_simple.yaml')

    model.train(
        data='data/dino_260417/dino_260417.yaml',
        epochs=300,
        imgsz=640,
        batch=20,
        device=device,
        workers=16,
        name='dino_simple_260425',
        exist_ok=True,
        amp=True,
        patience=50
    )


if __name__ == '__main__':
    main()
