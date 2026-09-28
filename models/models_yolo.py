# YOLO 모델 정의
"""
YOLO 학습/추론 스크립트.
 
main.py에서 `python main.py --model yolo`로 실행하면
이 파일의 run(project_root) 함수가 호출됩니다.
 
run() 함수 안에 실제 학습/추론 코드를 구현.
 
사용 가능한 데이터 (전처리 완료 후 기준):
    project_root/data/labels/data.yaml            (학습 설정 파일, train/val 경로 및 클래스 포함)
    project_root/data/labels/train_oversampled.txt (오버샘플링 반영된 학습 이미지 목록)
    project_root/data/labels/train/*.txt          (train 라벨, 이미지 1장당 1개)
    project_root/data/labels/val/*.txt            (val 라벨)
    project_root/data/labels/classes.txt           (0부터 시작하는 YOLO index 기준 이름 매핑)
    project_root/data/images/train, val, test           (이미지 원본)

result.py의 make_CSV 함수로 전달 하는 값 형태:
    sample_predictions = [
        {"image_id": 1, "category_id": 12778, "bbox": [x, y, w, h], "score": 0.91},
        {"image_id": 1, "category_id": 3743,  "bbox": [x, y, w, h], "score": 0.78},
        ...
    ]
        
    make_CSV(sample_predictions, "yolo", project_root)
"""
 
import os
import sys

# 이 파일을 "python models/models_yolo.py"처럼 직접 실행해도
# project_root(models 폴더의 상위 폴더)를 항상 찾을 수 있도록 경로를 보정한다.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models import result
from ultralytics import YOLO
import torch

def resolve_device(device: str = "auto") -> str:
    """device="auto"면 GPU 사용 가능 여부를 확인해서 '0'(GPU) 또는 'cpu'를 반환."""
    if device != "auto":
        return device
    return "0" if torch.cuda.is_available() else "cpu"

def load_category_id_table(classes_coco_path: str) -> list:
    """
    YOLO는 클래스를 0, 1, 2, ... 순서로 부르지만,
    원래 COCO 데이터셋의 category_id는 이 숫자와 다를 수 있음.
    예: YOLO의 클래스 0번이 실제로는 category_id=3 인 경우
 
    이 함수는 "YOLO 클래스 번호 -> 실제 category_id" 로 변환하는 표(리스트)를 만듬.
    data/classes_coco.txt 파일이 "category_id: 이름" 형식
    """
    category_id_table = []
    with open(classes_coco_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            category_id_str, _class_name = line.split(":", 1)
            category_id_table.append(int(category_id_str.strip()))
    return category_id_table

def train_model(data_yaml: str, epochs: int = 50, imgsz: int = 640, device: str = "auto") -> YOLO:
    """
    YOLOv8n(가장 작고 가벼운 버전) 사전학습 가중치를 불러와서,
    우리 데이터로 추가 학습(파인튜닝).
 
    device: "auto"(기본, GPU 있으면 자동 사용) / "0"(0번 GPU 지정) / "cpu"(강제로 CPU만 사용)

    "데이터 증강(augmentation)" 설정.
    """

    resolved_device = resolve_device(device)
    print(f"[YOLO] 사용 장치: {resolved_device}")

    model = YOLO("yolo11n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,

        # 배치사이즈 조정
        # 기존 auto 16에서 변경
        # batch=8,
 
        # 회전: 촬영 각도가 70/75/90도로 다양해서 보정용으로 사용
        degrees=10,
 
        # 밝기(hsv_v) / 색조(hsv_h) / 채도(hsv_s): test 이미지의 색 분포가 train보다 넓어서 여유를 둠
        hsv_v=0.5,
        hsv_h=0.02,
        hsv_s=0.6,
 
        # 좌우 반전: 각인 문자가 뒤집히면 못 알아볼 위험이 있어 확률을 낮게 설정
        fliplr=0.3,
        # 상하 반전: 사용하지 않음
        flipud=0.0,
 
        # 이동 / 크기 변화
        translate=0.1,
        scale=0.3,
 
        # Mosaic: 여러 이미지를 이어붙여 학습하는 기법.
        # 원본 이미지 하나에 이미 객체가 2~4개 있어서, 너무 많이 쓰면 오히려 혼란스러울 수 있어 줄임
        mosaic=0.5,

        # mixup: 두 이미지를 섞어서 학습 -> 모델이 더 일반화된 특징을 배우도록 함
        # copy_paste: 서로 다른 이미지의 객체를 복사해서 새로운 조합을 만들어 희귀 클래스 등장 빈도를 인위적으로 늘리는 효과
        # mixup=0.1,
        # copy_paste=0.15

    )
    return model
 
 
def predict_on_test_images(model: YOLO, images_test_dir: str, category_id_table: list, conf: float = 0.05) -> list:
    """
    결과를 Kaggle 제출 형식(csv)에 맞는 딕셔너리 리스트로 변환.
 
    conf: confidence(신뢰도) 임계값. 이 값보다 확신이 낮은 탐지 결과는 버림.
    """
    results = model.predict(source=images_test_dir, conf=conf, agnostic_nms=True, save=False, verbose=False)
 
    predictions = []
    for one_image_result in results:
        # test 파일명이 숫자로만 되어 있음 (예: "528.png" -> image_id=528)
        image_id = int(os.path.splitext(os.path.basename(one_image_result.path))[0])
 
        for box in one_image_result.boxes:
            yolo_class_idx = int(box.cls.item())
            confidence_score = float(box.conf.item())
 
            # 박스 좌표: (좌상단 x, 좌상단 y, 우하단 x, 우하단 y)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
 
            predictions.append({
                "image_id": image_id,
                "category_id": category_id_table[yolo_class_idx],
                "bbox": [x1, y1, width, height],
                "score": confidence_score,
            })
 
    return predictions

def run(project_root: str) -> None:
    print("[YOLO] 진입")


    """전체 파이프라인: 학습 -> 최고 성능 가중치로 추론 -> csv 생성"""
    data_yaml = os.path.join(project_root, "data", "labels", "data.yaml")
    classes_coco_path = os.path.join(project_root, "data", "classes_coco.txt")
    images_test_dir = os.path.join(project_root, "data", "images", "test")

    print(f"data_yaml = {data_yaml}")
    print(f"classes_coco_path = {classes_coco_path}")
    print(f"images_test_dir = {images_test_dir}")
 
    # 1. 학습
    print(f"[YOLO] 학습 시작 (data={data_yaml})")
    model = train_model(data_yaml)
 
    # 2. 학습 중 가장 성능이 좋았던 가중치(best.pt)를 불러와서 추론용으로 사용
    best_weights_path = model.trainer.best
    print(f"[YOLO] 학습 완료, best 가중치: {best_weights_path}")
    best_model = YOLO(best_weights_path)
 
    # 3. YOLO 클래스 번호 -> 실제 category_id 변환표 준비
    category_id_table = load_category_id_table(classes_coco_path)
 
    # 4. test 이미지 전체 추론
    print(f"[YOLO] test 추론 시작 ({images_test_dir})")
    predictions = predict_on_test_images(best_model, images_test_dir, category_id_table)
    print(f"[YOLO] 추론 완료, 예측 건수: {len(predictions)}")

    # 5. predictions 결과 값 확인
    # print(predictions)

    print("[YOLO] 종료")

    # 5. 제출용 csv 생성
    result.make_CSV(predictions, "yolo", project_root)
 
if __name__ == "__main__":
    # 단독 실행 테스트용 (project_root를 이 파일 기준 상위 폴더로 가정)
    run(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))