import cv2
import numpy as np
from .base_model import BasePredictor

class YOLOv8Predictor(BasePredictor):
    def __init__(self, weights_path: str = 'yolov8n.pt', device: str = 'cpu'):
        super().__init__(weights_path, device)
        self.classes = {}  # 클래스 매핑을 저장할 딕셔너리
    
    def load_model(self):
        """YOLOv8 모델 로드"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            print(f"YOLOv8 모델 로드 완료: {self.weights_path}")
        except ImportError:
            raise ImportError("ultralytics 패키지가 설치되어 있지 않습니다. pip install ultralytics")

    def predict(self, image_path: str, conf_thres=0.5, iou_thres=0.45) -> dict:
        """이미지 예측 수행"""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        # YOLOv8은 BGR 이미지를 직접 입력받습니다 (OpenCV는 BGR).
        # 단, 클래스 라벨링 시 RGB 기준이었다면 주의해야 합니다.
        
        results = self.model.predict(
            source=image_path,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False
        )
        
        # 결과 파싱
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()
            
            # 클래스 이름 매핑 (유니코딩/로딩된 모델의 경우)
            names = results[0].names
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls_id = int(cls_ids[i])
                class_name = names[cls_id]
                
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                })
        
        return {
            'image_path': image_path,
            'detections': detections
        }

    # 단일 알약 분류(Classification) 용도라면 YOLOv8 classify를 쓰는 것이 더 좋습니다.
    # 위 코드는 복수의 알약을 검출(Detection)하여 분류하는 것입니다.