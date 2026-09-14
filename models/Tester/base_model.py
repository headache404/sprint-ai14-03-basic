import cv2
from abc import ABC, abstractmethod

class BasePredictor(ABC):
    def __init__(self, weights_path: str = None, device: str = 'cpu'):
        self.weights_path = weights_path
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self):
        """모델 가중치를 로드합니다."""
        pass

    @abstractmethod
    def predict(self, image_path: str) -> dict:
        """이미지에 대해 예측을 수행합니다.
        
        Args:
            image_path (str): 예측할 이미지 경로
            
        Returns:
            dict: 예측 결과 (예: 클래스명, Confidence score 등)
        """
        pass

    def preprocess_image(self, image_path: str):
        """이미지를 전처리합니다."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        # YOLOv8은 BGR에서 RGB로 변환이 필요할 수 있음 ( 라이브러리 의존성 )
        # Faster R-CNN은 일반적으로 RGB를 기대하거나 ToTensor 과정에서 처리됨
        return img