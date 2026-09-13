# 모델의 결과를 제출용 CSV로 만든다


"""
Kaggle 제출용 csv 생성 모듈.
 
models_faster.py, models_yolo.py 양쪽에서 공통으로 불러와 사용합니다.
 
제출 형식 (mAP@[0.75:0.95] 채점 기준):
    annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
 
    [
        {"image_id": 1, "category_id": 12778, "bbox": [x, y, w, h], "score": 0.91},
        {"image_id": 1, "category_id": 3743,  "bbox": [x, y, w, h], "score": 0.78},
        ...
    ]

CSV 파일은 ../result 폴더에 생성 한다

네이밍룰 : 1423_모델명_일자_시분
            1423_faster_20260913_1030.csv
            1423_yolo_20260913_1030.csv
"""
 
import os
import csv

def make_CSV(project_root: str) -> str:
    print("make_CSV 호출")