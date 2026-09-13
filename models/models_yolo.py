# YOLO 모델 정의
"""
YOLO 학습/추론 스크립트.
 
main.py에서 `python main.py --model yolo`로 실행하면
이 파일의 run(project_root) 함수가 호출됩니다.
 
run() 함수 안에 실제 학습/추론 코드를 구현.
 
사용 가능한 데이터 (전처리 완료 후 기준):
    project_root/data/yolo_labels/data.yaml            (학습 설정 파일, train/val 경로 및 클래스 포함)
    project_root/data/yolo_labels/train_oversampled.txt (오버샘플링 반영된 학습 이미지 목록)
    project_root/data/yolo_labels/train/*.txt          (train 라벨, 이미지 1장당 1개)
    project_root/data/yolo_labels/val/*.txt            (val 라벨)
    project_root/data/yolo_labels/classes.txt           (0부터 시작하는 YOLO index 기준 이름 매핑)
    project_root/data/images/train, val, test           (이미지 원본)
"""
 
import os
from models import result

def run(project_root: str) -> None:
    """YOLO 학습/추론 구현 부분"""
    data_yaml = os.path.join(project_root, "data", "yolo_labels", "data.yaml")
    images_test = os.path.join(project_root, "data", "images", "test")
 
    print("[YOLO] 진입")
    print(f"  data_yaml   = {data_yaml}")
    print(f"  images_test = {images_test}")

    result.make_CSV(project_root)
 
if __name__ == "__main__":
    # 단독 실행 테스트용 (project_root를 이 파일 기준 상위 폴더로 가정)
    run(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))