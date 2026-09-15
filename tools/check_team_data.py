"""전처리 결과가 YOLO 학습에 연결 가능한지 읽기 전용으로 점검한다."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "data/yolo_labels/data.yaml",
    "data/yolo_labels/train_oversampled.txt",
    "data/yolo_labels/classes.txt",
    "data/coco_annotations/train.json",
    "data/coco_annotations/val.json",
    "data/images/train",
    "data/images/val",
    "data/images/test",
    "data/labels/train",
    "data/labels/val",
]


def image_stems(path: Path) -> set[str]:
    return {item.stem for item in path.glob("*.png")}


def label_stems(path: Path) -> set[str]:
    return {item.stem for item in path.glob("*.txt")}


def validate_label_file(path: Path, class_count: int) -> list[str]:
    errors = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number} 열 개수 {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path}:{line_number} 숫자 변환 실패")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{path}:{line_number} class_id={class_id}")
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            errors.append(f"{path}:{line_number} 좌표 범위 초과")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path}:{line_number} bbox 크기 오류")
    return errors


def main() -> None:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).exists()]
    if missing:
        print(json.dumps({"status": "MISSING", "missing": missing}, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    train_images = image_stems(ROOT / "data/images/train")
    val_images = image_stems(ROOT / "data/images/val")
    test_images = image_stems(ROOT / "data/images/test")
    train_labels = label_stems(ROOT / "data/labels/train")
    val_labels = label_stems(ROOT / "data/labels/val")
    classes = [
        line for line in (ROOT / "data/yolo_labels/classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    errors = []
    if train_images != train_labels:
        errors.append(f"train 이미지/라벨 이름 불일치: {len(train_images ^ train_labels)}개")
    if val_images != val_labels:
        errors.append(f"val 이미지/라벨 이름 불일치: {len(val_images ^ val_labels)}개")
    overlap = train_images & val_images
    if overlap:
        errors.append(f"train/val 중복: {len(overlap)}개")

    for label_dir in (ROOT / "data/labels/train", ROOT / "data/labels/val"):
        for label_path in label_dir.glob("*.txt"):
            errors.extend(validate_label_file(label_path, len(classes)))

    oversampled_paths = [
        Path(line.strip())
        for line in (ROOT / "data/yolo_labels/train_oversampled.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing_oversampled = [str(path) for path in oversampled_paths if not path.exists()]
    if missing_oversampled:
        errors.append(f"오버샘플링 이미지 경로 없음: {len(missing_oversampled)}개")

    report = {
        "status": "OK" if not errors else "FAILED",
        "counts": {
            "classes": len(classes),
            "train_images": len(train_images),
            "val_images": len(val_images),
            "test_images": len(test_images),
            "oversampled_train_entries": len(oversampled_paths),
        },
        "train_val_overlap": len(overlap),
        "errors": errors[:20],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
