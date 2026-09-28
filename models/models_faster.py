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


result.py의 make_CSV 함수로 전달 하는 값 형태:
    sample_predictions = [
        {"image_id": 1, "category_id": 12778, "bbox": [x, y, w, h], "score": 0.91},
        {"image_id": 1, "category_id": 3743,  "bbox": [x, y, w, h], "score": 0.78},
        ...
    ]
        
    make_CSV(sample_predictions, "faster", project_root)

필요 패키지: torch, torchvision, albumentations (pip install torch torchvision albumentations)

"""

import os
import sys
import json
from collections import defaultdict

# 이 파일을 "python models/models_faster.py"처럼 직접 실행해도
# project_root(models 폴더의 상위 폴더)를 항상 찾을 수 있도록 경로를 보정한다.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models import result
 
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------
class CocoPillDataset(Dataset):
    """train.json / val.json 형식의 COCO 라벨을 읽어 torchvision detection 모델 입력으로 변환한다."""
 
    def __init__(self, coco_json_path: str, images_dir: str, transform=None):
        with open(coco_json_path, encoding="utf-8") as f:
            coco = json.load(f)
 
        self.images_dir = images_dir
        self.transform = transform
        self.images_by_id = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images_by_id.keys())
 
        self.anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)
 
        # category_id(원본, 불연속) -> 모델 라벨(1부터 시작, 0은 배경 예약)
        cat_ids = sorted({c["id"] for c in coco["categories"]})
        self.cat_id_to_label = {cid: i + 1 for i, cid in enumerate(cat_ids)}
        self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}
 
    def __len__(self):
        return len(self.image_ids)
 
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_by_id[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = np.array(Image.open(img_path).convert("RGB"))
 
        boxes_coco, labels = [], []
        for ann in self.anns_by_image[img_id]:
            boxes_coco.append(ann["bbox"])  # [x, y, w, h]
            labels.append(self.cat_id_to_label[ann["category_id"]])
 
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes_coco, category_ids=labels)
            image = transformed["image"]
            boxes_coco = transformed["bboxes"]
            labels = transformed["category_ids"]
 
        # torchvision detection 모델은 [x1, y1, x2, y2] 형식을 기대한다.
        boxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes_coco]
 
        target = {
            "boxes": torch.as_tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return image, target
 
 
def collate_fn(batch):
    return tuple(zip(*batch))
 
 
# ---------------------------------------------------------------------------
# 증강 (train만 적용, val/test는 순수 변환만)
# ---------------------------------------------------------------------------
def build_train_transform() -> A.Compose:
    return A.Compose([
        A.Rotate(limit=10, p=0.5),                                    # 촬영각도 대응
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.5),  # test의 넓은 밝기 분포 대응
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.4),  # 배경색 다양성, hue는 약하게
        A.HorizontalFlip(p=0.3),                                       # 각인 문자 고려해 낮은 확률
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], min_visibility=0.3))
 
 
def build_eval_transform() -> A.Compose:
    return A.Compose([
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]))
 
 
# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def build_model(num_classes: int):
    """ImageNet/COCO 사전학습된 Faster R-CNN(ResNet50 FPN 백본)의 head만 우리 클래스 수에 맞게 교체한다."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=600,     # 기본 800 -> 480으로 낮춤 (메모리 절약) -> 600
        max_size=800      # 기본 1333 -> 640으로 낮춤 -> 800
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # num_classes + 1: 배경(0) 클래스를 포함해야 함
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
 
 
def train_model(train_dataset, val_dataset, epochs: int = 50, batch_size: int = 8, lr: float = 0.002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Faster R-CNN] 사용 장치: {device}")
 
    num_classes = len(train_dataset.cat_id_to_label)
    print("모델 생성 및 GPU 로드 중...")
    model = build_model(num_classes).to(device)
    print("모델 GPU 로드 완료")
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
 
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            epoch_loss += loss.item()
 
        lr_scheduler.step()
        print(f"[Faster R-CNN] epoch {epoch + 1}/{epochs} - train loss: {epoch_loss / len(train_loader):.4f}")
 
    return model, device
 
 
# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------
def predict_test(model, device, images_test_dir: str, label_to_cat_id: dict,
                  eval_transform: A.Compose, score_threshold: float = 0.05) -> list:
    model.eval()
    predictions = []
 
    for fname in sorted(os.listdir(images_test_dir)):
        if not fname.lower().endswith(".png"):
            continue
 
        image_id = int(os.path.splitext(fname)[0])
        img_path = os.path.join(images_test_dir, fname)
        image_np = np.array(Image.open(img_path).convert("RGB"))
 
        transformed = eval_transform(image=image_np, bboxes=[], category_ids=[])
        image_tensor = transformed["image"].unsqueeze(0).to(device)
 
        with torch.no_grad():
            outputs = model(image_tensor)[0]
 
        boxes = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
 
        for box, label, score in zip(boxes, labels, scores):
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = box
            predictions.append({
                "image_id": image_id,
                "category_id": label_to_cat_id[int(label)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
 
    return predictions
 
# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def run(project_root: str) -> None:
    """Faster R-CNN 학습 -> test 추론 -> Kaggle 제출용 csv 생성까지 전체를 수행한다."""
    train_json = os.path.join(project_root, "data", "coco_annotations", "train.json")
    val_json = os.path.join(project_root, "data", "coco_annotations", "val.json")
    images_train = os.path.join(project_root, "data", "images", "train")
    images_val = os.path.join(project_root, "data", "images", "val")
    images_test = os.path.join(project_root, "data", "images", "test")
 
    print("[Faster R-CNN] 데이터셋 로딩")
    train_dataset = CocoPillDataset(train_json, images_train, transform=build_train_transform())
    val_dataset = CocoPillDataset(val_json, images_val, transform=build_eval_transform())
 
    print(f"[Faster R-CNN] train {len(train_dataset)}개 / val {len(val_dataset)}개 / 클래스 {len(train_dataset.cat_id_to_label)}개")
 
    model, device = train_model(train_dataset, val_dataset)
 
    print(f"[Faster R-CNN] test 추론 시작 ({images_test})")
    predictions = predict_test(
        model, device, images_test,
        label_to_cat_id=train_dataset.label_to_cat_id,
        eval_transform=build_eval_transform(),
    )
    print(f"[Faster R-CNN] 추론 완료, 예측 건수: {len(predictions)}")
 
    result.make_CSV(predictions, "faster", project_root)

if __name__ == "__main__":
    # 단독 실행 테스트용 (project_root를 이 파일 기준 상위 폴더로 가정)
    run(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))