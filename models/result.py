# 모델의 결과를 제출용 CSV로 만든다


"""
Kaggle 제출용 csv 생성 모듈.
 
models_faster.py, models_yolo.py 양쪽에서 공통으로 불러와 사용합니다.
 
제출 형식 (mAP@[0.75:0.95] 채점 기준):
    annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score

각 모델에서 make_CSV 함수로 전달 하는 값 형태:
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
# import re
import datetime
import pandas as pd
# import matplotlib.pyplot as plt

def make_CSV(predictions: list, model_name: str, project_root: str, output_dir: str = "result") -> str:
    """
    예측 데이터 양식에 맞춰 캐글 제출용 CSV를 생성합니다.
    입력 양식 예시: [{'image_id': 1, 'category_id': 12778, 'bbox': [x, y, w, h], 'score': 0.91}, ...]
    네이밍룰: 1423_모델명_일자_시분.csv (예: 1423_yolo_20260916_1430.csv)
    """
    rows = []
    
    for idx, pred in enumerate(predictions, start=1):
        image_id = pred.get('image_id', 0)
        category_id = pred.get('category_id', 0)
        bbox = pred.get('bbox', [0, 0, 0, 0])  # [x, y, w, h] 구조
        score = pred.get('score', 0.0)
        
        # bbox 데이터 분해
        x, y, w, h = bbox
        
        rows.append({
            'annotation_id': idx,
            'image_id': int(image_id),
            'category_id': int(category_id),
            'bbox_x': round(float(x), 2),
            'bbox_y': round(float(y), 2),
            'bbox_w': round(float(w), 2),
            'bbox_h': round(float(h), 2),
            'score': round(float(score), 4)
        })

    # 캐글 필수 컬럼 순서 지정
    columns = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
    df = pd.DataFrame(rows, columns=columns)
    
    # ../result 폴더에 네이밍룰 적용하여 저장
    out_dir = os.path.join(project_root, output_dir)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f"1423_{model_name.lower()}_{timestamp}.csv"
    csv_path = os.path.join(out_dir, file_name)
    
    df.to_csv(csv_path, index=False)
    print(f"[{model_name}] 캐글 제출용 CSV 저장 완료: {csv_path} (총 객체 수: {len(df)}개)")
    return csv_path

'''
# visualize는 일단 주석으로 처리 함
def visualize_result_csv(result_csv_path: str):
    """
    팀원이 준 results.csv 파일을 읽어 100 에포크 동안의 Loss와 mAP 변화를 시각화합니다.
    """
    if not os.path.exists(result_csv_path):
        print(f"[에러] results.csv 파일을 찾을 수 없습니다: {result_csv_path}")
        return

    # results.csv 파일 불러오기
    df = pd.read_csv(result_csv_path)
    df.columns = df.columns.str.strip()

    print(f"[결과 분석] {result_csv_path} 파일 로드 완료 (총 {len(df)} 에포크)")
    print(df.head())

    # 시각화 그래프 설정 (Loss 와 mAP)
    plt.figure(figsize=(14, 5))

    # 1. Loss 그래프 (Train Loss / Val Loss 등)
    plt.subplot(1, 2, 1)
    loss_cols = [c for c in df.columns if 'loss' in c.lower()]
    for col in loss_cols:
        plt.plot(df['epoch'], df[col], label=col)
    plt.title("Training & Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. 성능 지표 (mAP 등) 그래프
    plt.subplot(1, 2, 2)
    metric_cols = [c for c in df.columns if 'map' in c.lower() or 'precision' in c.lower() or 'recall' in c.lower()]
    for col in metric_cols:
        plt.plot(df['epoch'], df[col], label=col)
    plt.title("Metrics (mAP, Precision, Recall)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
'''

'''
if __name__ == "__main__":
    # 1. results.csv 파일을 불러와서 시각화 실행하는 부분
    # RESULT_CSV_PATH = r"C:\Users\home\OneDrive\Desktop\코드잇\sprint-ai14-03-basic\results_log\results.csv"
    # visualize_result_csv(RESULT_CSV_PATH)
    
    # 2. 양식에 맞춘 캐글 제출용 CSV 생성 함수 테스트 예시
    sample_predictions = [
        {"image_id": 1, "category_id": 12778, "bbox": [100.5, 150.2, 50.1, 80.4], "score": 0.91},
        {"image_id": 1, "category_id": 3743,  "bbox": [300.0, 350.0, 40.2, 45.1], "score": 0.78}
    ]
    
    # 모델명과 함께 리스트화하여 제출용 CSV 생성 확인
    make_CSV(sample_predictions, "yolo", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
'''