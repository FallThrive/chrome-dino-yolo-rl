from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ultralytics import YOLO
import supervision as sv


@dataclass
class Detection:
    label: str
    x_left: float
    y_top: float
    x_right: float
    y_bottom: float
    confidence: float
    x_center: float = 0.0
    y_center: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    def __post_init__(self):
        self.x_center = (self.x_left + self.x_right) / 2
        self.y_center = (self.y_top + self.y_bottom) / 2
        self.width = self.x_right - self.x_left
        self.height = self.y_bottom - self.y_top


@dataclass
class DinoDetectionResult:
    dino: Optional[Detection]
    obstacles: List[Detection]
    has_restart: bool
    raw_detections: sv.Detections
    image: np.ndarray


class DinoDetector:
    DINO_X_POSITION = 40
    
    def __init__(self, model_path: str = "weights/yolo26n_dino_simple.pt", confidence: float = 0.6):
        self.model = YOLO(model_path)
        self.confidence = confidence
    
    def detect(self, image: np.ndarray) -> DinoDetectionResult:
        results = self.model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > self.confidence]
        
        dino = None
        obstacles = []
        has_restart = False
        
        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i]
            class_name = detections.data['class_name'][i]
            confidence = detections.confidence[i]
            
            detection = Detection(
                label=class_name,
                x_left=box[0],
                y_top=box[1],
                x_right=box[2],
                y_bottom=box[3],
                confidence=confidence
            )
            
            if class_name == 'dino':
                dino = detection
            elif class_name in ['cactus', 'bird']:
                obstacles.append(detection)
            elif class_name == 'restart':
                has_restart = True
        
        obstacles.sort(key=lambda x: x.x_left)
        
        return DinoDetectionResult(
            dino=dino,
            obstacles=obstacles,
            has_restart=has_restart,
            raw_detections=detections,
            image=image
        )
    
    def get_obstacles_right_of_dino(self, obstacles: List[Detection], dino_x: float = DINO_X_POSITION) -> List[Detection]:
        return [obs for obs in obstacles if obs.x_left > dino_x]
    
    def has_label(self, detections: sv.Detections, label_name: str) -> bool:
        if len(detections) == 0:
            return False
        class_names = detections.data.get('class_name', [])
        return label_name in class_names
