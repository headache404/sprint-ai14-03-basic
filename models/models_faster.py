# Faster R-CNN 모델 정의
"""
Faster R-CNN 학습/추론 스크립트.

main.py에서 `python main.py --model faster`로 실행하면
이 파일의 run(project_root) 함수가 호출됩니다.

run() 함수 안에 실제 학습/추론 코드를 구현.

사용 가능한 데이터 (전처리 완료 후 기준):
    project_root/data/coco_annotations/train.json   (Faster R-CNN용 COCO 라벨, images 311.. / annotations 1105..)
    project_root/data/coco_annotations/val.json
    project_root/data/images/train/                 (원본 이미지, 오버샘플링은 JSON의 image_id로만 반영됨)
    project_root/data/images/val/
    project_root/data/images/test/                  (라벨 없음, 최종 예측용)
    project_root/data/classes_coco.txt               (category_id: 이름 매핑, 참고용)
"""

import os
import sys

# 이 파일을 "python models/models_faster.py"처럼 직접 실행해도
# project_root(models 폴더의 상위 폴더)를 항상 찾을 수 있도록 경로를 보정한다.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models import result

def run(project_root: str) -> None:
    """Faster R-CNN 학습/추론 구현 부분"""
    train_json = os.path.join(project_root, "data", "coco_annotations", "train.json")
    val_json = os.path.join(project_root, "data", "coco_annotations", "val.json")
    images_train = os.path.join(project_root, "data", "images", "train")
    images_val = os.path.join(project_root, "data", "images", "val")
    images_test = os.path.join(project_root, "data", "images", "test")

    print("[Faster R-CNN] 진입")
    print(f"  train_json  = {train_json}")
    print(f"  val_json    = {val_json}")
    print(f"  images_train= {images_train}")
    print(f"  images_val  = {images_val}")
    print(f"  images_test = {images_test}")

    result.make_CSV(project_root)

if __name__ == "__main__":
    # 단독 실행 테스트용 (project_root를 이 파일 기준 상위 폴더로 가정)
    run(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))