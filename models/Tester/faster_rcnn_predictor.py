import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from .base_model import BasePredictor

class FasterRCNNDetector(BasePredictor):
    def __init__(self, weights_path: str = None, device: str = 'cpu'):
        super().__init__(weights_path, device)
        self.num_classes = 0 # 훈련 시 정의된 클래스 수
        
    def load_model(self):
        """Faster R-CNN 모델 로드"""
        try:
            # 사전 학습된 가중치 사용 (COCO 데이터셋 기반)
            # 알약 분류를 위해 파인튜닝된 가중치가 있다면 그 경로를 weights_path에 넣으세요.
            self.model = fasterrcnn_resnet50_fpn(pretrained=True) 
            self.model.to(self.device)
            
            # 만약 파인튜닝된 가중치가 있다면 다음 주석 해제 및 경로 설정
            # if self.weights_path:
            #     state_dict = torch.load(self.weights_path, map_location=self.device)
            #     self.model.load_state_dict(state_dict)
            
            self.model.eval()
            print(f"Faster R-CNN 모델 로드 완료 (Pretrained COCO)")
        except Exception as e:
            raise RuntimeError(f"Faster R-CNN 모델 로드 실패: {e}")

    @staticmethod
    def transform(image, target):
        """이미지를 tensor로 변환하고 정규화"""
        image = F.to_tensor(image)
        # Faster R-CNN 입력은 Normalize 필요 없으므로 여기서는 단순 to_tensor 사용
        # 실제 훈련 시에는 Mean, Std에 맞춰야 함
        return image

    def predict(self, image_path: str) -> dict:
        """이미지 예측 수행"""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        # 이미지 읽기
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        
        # RGB 변환 (OpenCV는 BGR, PyTorch는 일반적으로 RGB 기대)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tensor 변환
        image_tensor = F.to_tensor(img_rgb).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([image_tensor])
            
        detections = []
        # 예측 결과 파싱 (COCO 기준 클래스 ID는 알약 클래스와 다름. 파인튜닝 시 수정 필요)
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] > 0.5: # 신뢰도 임계값
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    'class_id': int(labels[i]),
                    'confidence': float(scores[i]),
                    'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                })

        return {
            'image_path': image_path,
            'detections': detections
        }