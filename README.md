# 🚀 경구약제 이미지 객체 검출(Object Detection) 프로젝트
 
이 프로젝트는 Sprint AI14기 Part2 3팀의 Basic Project 입니다  
이번 프로젝트의 목표는 사진 속에 있는 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출하는 것입니다.

## 👥 멤버

🏷️ 팀명 **1423**
| 이름 | 역할 | GitHub |
|------|------|--------|
| 서동현 | Project Leader | [@github](https://github.com/headache404) |
| 김원태 | Experimentation Lead | [@github](https://github.com/andyKim0313) |
| 김주희 | Data Engineer | [@github](https://github.com/juhee4839) |
| 이석우 | Experimentation Lead | [@github](https://github.com/SUKWOOLEE-249) |
| 조영권 | Model Architect | [@github](https://github.com/Young9won) |

---
 
## 📌 프로젝트 설명
 
- **이미지 인식 기술을 헬스케어 분야에 접목해보는** 프로젝트 입니다.
- 헬스케어 스타트업 : 헬스잇(Health Eat) 의 AI 엔지니어링 팀이라고 가정 합니다.
    - AI 엔지니어링 팀은 유저가 본인의 모바일 애플리케이션으로 자신이 복용중인 약 사진을 찍었을 때, 이미지 인식을 통해 해당 약에 대한 정보를 확인할 수 있는 모델을 만들어야하는 미션을 부여받았습니다.
    - 기업에서는 이를 통해 유저의 건강 상태 및 함께 복용하면 안되는 약 등 헬스케어 정보를 유저들에게 제공 합니다.
- 사진 속에 있는 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출하는 모델을 구현하고, 성능을 지속적으로 개선해나가는 것이 프로젝트의 목표입니다.

---

## 📂 프로젝트 구조
 
```
project/
├── data/
│   ├── images/                     # train, val, test image
│   ├── labels/                     # Ultralytics YOLO 라벨 (train, val)
│   ├── yolo_labels/                # data.yaml, 클래스 및 오버샘플링 목록
│   ├── coco_annotations/           # train, val json
├── models/                         # 모델 정의
├── notebooks/                      # 데이터 탐색을 위한 노트북
├── result/                         # 모델 결과 CSV 파일
├── utils/                          # 데이터 로딩 유틸리티
├── main.py                         # 메인 실행 스크립트
├── environment.yml                 # conda 설치 패키지 목록
├── requirements.colab.txt          # Colab에서 추가 설치할 패키지
└── README.md
```
 
## ⚙️ 실행 방법
 
**1. 필요한 패키지 설치**
```bash
pip install torch torchvision
```
 
**2. 최초 데이터 전처리**
```bash
python main.py --preprocess --model none
```

**3. YOLO 연결 시험**
```bash
python main.py --skip --model yolo --yolo-smoke
```

**4. YOLOv8n baseline 학습 및 평가**
```bash
python main.py --skip --model yolo
```

로컬에 CUDA를 지원하는 NVIDIA GPU가 없으면
`notebooks/03_YOLO_Colab_GPU_학습.ipynb`를 Colab에서 열고 위에서부터 실행합니다.
노트북은 GPU 확인, Kaggle 데이터 다운로드, 전처리, 시험 학습, baseline 학습과 결과 보존을 안내합니다.

학습 후 test 이미지의 YOLO 형식 예측 라벨까지 만들려면 옵션을 추가합니다.
```bash
python main.py --skip --model yolo --yolo-predict-test
```

---
 
## 📊 결과
 
 스크린샷, 출력 예시, 성능 지표(정확도, F1-score 등)를 표나 이미지.
 
| 모델 | 정확도 | F1-score |
|------|--------|----------|
| Baseline | - | - |
| Fine-tuned | - | - |

- [보고서](https://github.com/github_id)

---
 
## 📋 협업일지
 
| 이름 | URL |
|------|--------|
| 서동현 | [URL](https://github.com/github_id) |
| 김원태 | [URL](https://github.com/github_id) |
| 김주희 | [URL](https://github.com/github_id) |
| 이석우 | [URL](https://github.com/github_id) |
| 조영권 | [URL](https://github.com/github_id) |
 
---
 
## 📝 참고 사항
 
- Image, json, txt, csv 파일은 별도 업로드 하지 않습니다
