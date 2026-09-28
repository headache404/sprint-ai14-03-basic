"""
train.json의 bbox 이상치(이미지 범위 초과, 0 이하 크기 등)를 제거하고,
YOLO 라벨(train+val)과 오버샘플링 목록을 다시 생성한 뒤 캐시까지 삭제하는 스크립트.

merge_aihub_class.py에서 이미 만든 함수들(regenerate_yolo_labels, update_oversampled_list,
delete_yolo_cache)을 그대로 재사용한다.

사용법 (프로젝트 루트에서 실행):
    python clean_bbox.py
"""

import os
import json

from merge_aihub_class import (
    COCO_TRAIN_JSON,
    COCO_VAL_JSON,
    IMAGES_TRAIN_DIR,
    LABELS_TRAIN_DIR,
    TARGET_MIN,
    regenerate_yolo_labels,
    update_oversampled_list,
    delete_yolo_cache,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def is_valid_bbox(ann: dict, img_by_id: dict) -> bool:
    """bbox가 존재하고, 이미지 범위 안에 있으며, 크기가 0보다 큰지 확인한다."""
    img = img_by_id[ann["image_id"]]
    img_w, img_h = img["width"], img["height"]

    bbox = ann.get("bbox")
    if bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
        return False

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
        return False
    return True


def clean_train_bbox(coco_train_json_path: str) -> dict:
    """train.json에서 잘못된 bbox의 annotation을 제거하고, annotation_id를 재부여한 뒤 저장한다."""
    with open(coco_train_json_path, encoding="utf-8") as f:
        train = json.load(f)

    img_by_id = {img["id"]: img for img in train["images"]}

    before = len(train["annotations"])
    train["annotations"] = [a for a in train["annotations"] if is_valid_bbox(a, img_by_id)]
    after = len(train["annotations"])

    for i, ann in enumerate(train["annotations"], start=1):
        ann["id"] = i

    with open(coco_train_json_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    print(f"[bbox 정제] 제거 전: {before} / 제거 후: {after} / 제거된 개수: {before - after}")
    return train


def find_label_count_mismatch(train: dict) -> list:
    """파일명의 조합 성분 개수(K-a-b-c-d)와 실제 annotation 개수가 다른 이미지를 찾는다.

    조합 이미지는 원래 file_name 안의 "K-" 코드 개수(예: K-a-b-c-d -> 4개)만큼
    annotation이 있어야 한다. 개수가 안 맞으면 라벨 누락/중복이 원본 데이터에
    있었다는 신호이므로, 부족한 경우(under)는 학습에서 제외한다.
    """
    anns_by_image = {}
    for ann in train["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    mismatches = []
    for img in train["images"]:
        combo_part = img["file_name"].split("_")[0]
        codes = combo_part.split("-")[1:]
        expected_count = len(codes)
        actual_count = len(anns_by_image.get(img["id"], []))

        if expected_count != actual_count:
            mismatches.append({
                "file_name": img["file_name"],
                "expected": expected_count,
                "actual": actual_count,
            })

    return mismatches


def remove_label_count_mismatch(train: dict) -> dict:
    """라벨 개수가 파일명의 조합 성분 수보다 부족한(under) 이미지를 train에서 제외한다.
    (과다한 경우(over)는 원본 데이터의 다른 문제일 수 있어 이번 단계에서는 건드리지 않는다)
    """
    mismatches = find_label_count_mismatch(train)
    under = [m for m in mismatches if m["actual"] < m["expected"]]
    over = [m for m in mismatches if m["actual"] > m["expected"]]
    under_fnames = {m["file_name"] for m in under}

    print(f"  라벨 불일치 총 {len(mismatches)}건 (부족 {len(under)}건, 과다 {len(over)}건)")
    print(f"  부족한 {len(under)}건은 이미지 자체를 제외, 과다 {len(over)}건은 이번 단계에서 유지")

    before_images = len(train["images"])
    before_anns = len(train["annotations"])

    train["images"] = [img for img in train["images"] if img["file_name"] not in under_fnames]
    valid_image_ids = {img["id"] for img in train["images"]}
    train["annotations"] = [a for a in train["annotations"] if a["image_id"] in valid_image_ids]

    for i, ann in enumerate(train["annotations"], start=1):
        ann["id"] = i

    print(f"  이미지: {before_images} -> {len(train['images'])} (제외 {before_images - len(train['images'])}개)")
    print(f"  annotation: {before_anns} -> {len(train['annotations'])}")
    return train


# 라벨 불일치("파일명 조합 성분 수" != "실제 annotation 수") 이미지를 제거할지 여부.
# 실제로 시각화해보니, 라벨이 일부만 붙어있고 나머지 알약엔 원래부터 라벨이 없는
# 이미지가 많았다(=AI Hub 원본이 의도적으로 일부만 라벨링한 것으로 추정). 이런 경우
# 이미지를 통째로 버리면 "라벨이 있는 나머지 물체"의 학습 기회까지 같이 사라지므로,
# 기본값은 False(적용 안 함)로 둠.
REMOVE_LABEL_MISMATCH = False


def main():
    print("=" * 70)
    print("1. train.json bbox 이상치 제거")
    print("=" * 70)
    coco_train = clean_train_bbox(COCO_TRAIN_JSON)

    if REMOVE_LABEL_MISMATCH:
        print("\n" + "=" * 70)
        print("2. 라벨 불일치(조합 성분 수 vs 실제 annotation 수) 제거")
        print("=" * 70)
        coco_train = remove_label_count_mismatch(coco_train)
        with open(COCO_TRAIN_JSON, "w", encoding="utf-8") as f:
            json.dump(coco_train, f, ensure_ascii=False, indent=2)
        print("  train.json 저장 완료")
    else:
        print("\n" + "=" * 70)
        print("2. 라벨 불일치 제거 - 건너뜀 (REMOVE_LABEL_MISMATCH=False, 실험 결과 역효과 확인됨)")
        print("=" * 70)
        find_label_count_mismatch_result = find_label_count_mismatch(coco_train)
        print(f"  참고: 여전히 {len(find_label_count_mismatch_result)}건의 불일치가 있지만 제거하지 않음")

    print("\n" + "=" * 70)
    print("3. YOLO 라벨 재생성 (train + val 동일 클래스 인덱스로)")
    print("=" * 70)
    with open(COCO_VAL_JSON, encoding="utf-8") as f:
        coco_val = json.load(f)
    labels_val_dir = os.path.join(os.path.dirname(LABELS_TRAIN_DIR), "val")
    regenerate_yolo_labels(coco_train, LABELS_TRAIN_DIR, coco_val=coco_val, labels_val_dir=labels_val_dir)

    print("\n" + "=" * 70)
    print("4. 오버샘플링 목록(train_oversampled.txt) 갱신")
    print("=" * 70)
    update_oversampled_list(coco_train, IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR, target_min=TARGET_MIN)

    print("\n" + "=" * 70)
    print("5. YOLO 캐시 삭제")
    print("=" * 70)
    delete_yolo_cache(LABELS_TRAIN_DIR)

    print("\n완료. val.json은 건드리지 않았습니다 (기존 검증 기준 유지).")


if __name__ == "__main__":
    main()