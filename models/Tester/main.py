import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 현재 디렉토리 경로 추가

from models.Tester import YOLOv8Predictor, FasterRCNNDetector

# 1. YOLOv8 예측기 초기화 및 모델 로드
yolo_predictor = YOLOv8Predictor(weights_path='path/to/your_alpaca_yolov8.pt', device='cpu') # gpu 가능하면 'cuda'
yolo_predictor.load_model()

# 2. Faster R-CNN 예측기 초기화 및 모델 로드
rcnn_predictor = FasterRCNNDetector(weights_path=None, device='cpu')
rcnn_predictor.load_model()

# 3. 이미지 경로
test_image_path = 'data/your_test_image.jpg'

# 4. 추론 수행
yolo_result = yolo_predictor.predict(test_image_path)
print("YOLOv8 Result:", yolo_result)

rcnn_result = rcnn_predictor.predict(test_image_path)
print("Faster R-CNN Result:", rcnn_result)