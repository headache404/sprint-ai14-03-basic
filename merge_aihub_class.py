"""
AI Hub 데이터에서, 기존 프로젝트 클래스와 이름이 겹치지 않는 모든 새 클래스를
기존 프로젝트의 COCO/YOLO 산출물에 통째로 추가 병합하는 스크립트 

전제:
    - 기존 전처리(python main.py --preprocess)가 이미 한 번 실행되어
      data/coco_annotations/train.json, val.json, data/images/train, data/labels/train 등이 만들어져 있어야 함.
    - 아래 AIHUB_SETS에 적힌 라벨/이미지 경로들이 실제로 존재해야 함.

사용법 (프로젝트 루트에서 실행):
    python merge_aihub_class.py
"""

import os
import json
import glob
import math
import shutil
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# AI Hub 원본 경로들 - 여러 세트를 한 번에 처리
AIHUB_SETS = [
    {
        "label_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_annotations", "TL_1"),
        "image_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_images", "TS_1"),
    },
    {
        "label_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_annotations", "TL_3"),
        "image_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_images", "TS_3"),
    },
    {
        "label_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_annotations", "TL_4"),
        "image_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_images", "TS_4"),
    },
    {
        "label_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_annotations", "TL_5"),
        "image_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_images", "TS_5"),
    },
    {
        "label_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_annotations", "TL_6"),
        "image_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_images", "TS_6"),
    },
    {
        "label_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_annotations", "TL_7"),
        "image_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_images", "TS_7"),
    },
    {
        "label_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_annotations", "TL_8"),
        "image_dir": os.path.join(PROJECT_ROOT, "sprint_ai_hub", "train_images", "TS_8"),
    },
]

# 기존 프로젝트 산출물 경로
COCO_TRAIN_JSON = os.path.join(PROJECT_ROOT, "data", "coco_annotations", "train.json")
COCO_VAL_JSON = os.path.join(PROJECT_ROOT, "data", "coco_annotations", "val.json")
IMAGES_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
LABELS_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "labels", "train")
CLASSES_COCO_TXT = os.path.join(PROJECT_ROOT, "data", "classes_coco.txt")
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "labels", "data.yaml")

# "all": 기존 클래스와 이름이 안 겹치는 모든 새 클래스를 통째로 추가 (B안)
# "target": TARGET_DL_IDX에 적은 클래스만 선별 추가
MODE = "all"
TARGET_DL_IDX = [12419]  # MODE="target"일 때만 사용

# 오버샘플링 목표값 (기존 main.py --target-min 기본값과 동일하게 맞춰야 비교가 공정함)
TARGET_MIN = 9


def build_image_path_index(image_dir: str) -> dict:
    """image_dir 하위(여러 단계의 폴더 포함)에 있는 모든 png 파일을 훑어,
    file_name -> 실제 전체 경로로 매핑한 딕셔너리를 만든다.
    (TS_3 폴더 바로 아래가 아니라, K-xxx-xxx-xxx 하위 폴더 안에 이미지가 있는 구조 대응)
    """
    index = {}
    for path in glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True):
        index[os.path.basename(path)] = path
    print(f"  이미지 인덱스 구축 완료: {len(index)}개 파일 ({image_dir})")
    return index


def load_existing_name_to_id(classes_coco_path: str) -> dict:
    """기존 프로젝트 classes_coco.txt에서 '이름 -> category_id' 매핑을 만든다.
    AI Hub의 dl_idx가 기존 클래스와 이름이 같으면, 새 id 대신 기존 id로 통일하기 위함
    (AI Hub와 우리 프로젝트의 dl_idx 체계가 서로 오프셋이 달라, 같은 약이 다른 숫자를
    갖는 문제를 막는다. 예: 무코스타정이 기존=3544, AI Hub=3543 처럼 1씩 다른 경우).
    """
    name_to_id = {}
    if not os.path.exists(classes_coco_path):
        print(f"  [경고] {classes_coco_path} 이 없어 이름 매칭을 건너뜁니다.")
        return name_to_id
    with open(classes_coco_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid_str, name = line.split(":", 1)
            name_to_id[name.strip()] = int(cid_str.strip())
    return name_to_id


def resolve_category_id(dl_idx: int, dl_name: str, existing_name_to_id: dict) -> int:
    """이름이 기존 클래스와 일치하면 기존 category_id를 쓰고, 아니면 AI Hub의 dl_idx를 그대로 쓴다."""
    return existing_name_to_id.get(dl_name.strip(), dl_idx)


def merge_all_components(sets: list, existing_name_to_id: dict):
    """여러 개의 (label_dir, image_dir) 세트를 전부 file_name 기준으로 병합한다.

    조합 이미지 하나는 여러 JSON(성분별)으로 쪼개져 있고, 각 JSON은 자기 자신의
    dl_idx만 담고 있다(annotations.category_id=1 고정, 실제 클래스는 images.dl_idx).
    따라서 여기서는 각 JSON의 annotation에 "그 JSON 자신의 dl_idx"를 category_id로
    부여하며 file_name 기준으로 합친다 (원래 sprint_ai_project1_data를 병합했던 방식과 동일).

    이때 dl_name이 기존 클래스 중 하나와 일치하면, AI Hub의 dl_idx가 아니라
    기존 category_id로 강제 통일한다 (오프셋 차이로 같은 약이 다른 클래스가 되는 것 방지).
    """
    merged = defaultdict(lambda: {"image_info": None, "annotations": [], "image_path": None})
    dl_idx_to_name = {}
    total_invalid_bbox = 0
    missing_images = 0
    remapped_count = 0

    for s in sets:
        label_dir, image_dir = s["label_dir"], s["image_dir"]
        print(f"  -- 세트 처리 중: {label_dir}")
        json_files = glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True)
        image_index = build_image_path_index(image_dir)

        for jf in json_files:
            try:
                with open(jf, encoding="utf-8") as f:
                    d = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            img_info = d["images"][0]
            raw_dl_idx = int(str(img_info["drug_N"]).replace("K-", "").lstrip("0") or "0") # raw_dl_idx = int(img_info["dl_idx"])
            dl_name = img_info.get("dl_name", "")

            real_cid = resolve_category_id(raw_dl_idx, dl_name, existing_name_to_id)
            if real_cid != raw_dl_idx:
                remapped_count += 1

            dl_idx_to_name[real_cid] = dl_name

            fname = img_info["file_name"]
            img_path = image_index.get(fname)
            if img_path is None:
                missing_images += 1
                continue

            entry = merged[fname]
            if entry["image_info"] is None:
                entry["image_info"] = img_info
                entry["image_path"] = img_path

            for ann in d["annotations"]:
                bbox = ann.get("bbox")
                if bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
                    total_invalid_bbox += 1
                    continue
                new_ann = dict(ann)
                new_ann["category_id"] = real_cid  # 이름이 같으면 기존 id로, 아니면 AI Hub dl_idx로
                entry["annotations"].append(new_ann)

    if total_invalid_bbox:
        print(f"  [경고] bbox 형식이 잘못되어 건너뛴 annotation {total_invalid_bbox}개")
    if missing_images:
        print(f"  [경고] 이미지 파일을 찾지 못한 JSON {missing_images}개")
    if remapped_count:
        print(f"  [정보] 이름 일치로 기존 category_id로 재매핑된 JSON {remapped_count}개")

    print(f"  전체 병합된 고유 이미지 수: {len(merged)}, 고유 클래스 수: {len(dl_idx_to_name)}")
    return merged, dl_idx_to_name


def filter_new_classes_only(merged: dict, dl_idx_to_name: dict, existing_ids: set):
    """merged(전체 병합 결과) 중, annotation이 전부 기존 클래스로만 이루어진 이미지는 제외하고,
    최소 1개의 '새로운(기존에 없는) 클래스'가 포함된 이미지만 남긴다.
    (그 이미지의 기존 클래스 성분도 그대로 유지 — 라벨 누락 방지, filter_images_containing_targets와 동일한 원리)
    """
    filtered = {}
    for fname, entry in merged.items():
        cats_in_image = {ann["category_id"] for ann in entry["annotations"]}
        if cats_in_image - existing_ids:  # 기존에 없는 클래스가 최소 1개 포함
            filtered[fname] = entry
    return filtered


def filter_images_containing_targets(merged: dict, target_ids: list):
    """merged(전체 병합 결과) 중, target_ids 중 하나라도 포함된 이미지만 골라낸다.
    (해당 이미지의 다른 성분 annotation도 전부 그대로 유지한다 — 라벨 누락 방지)
    """
    target_set = set(target_ids)
    filtered = {}
    for fname, entry in merged.items():
        cats_in_image = {ann["category_id"] for ann in entry["annotations"]}
        if cats_in_image & target_set:
            filtered[fname] = entry
    return filtered


def append_to_coco_json(coco_json_path: str, new_merged: dict, matched_names: dict, images_src_dir: str, images_dst_dir: str):
    """기존 train.json(또는 val.json)에 새 이미지/annotation/category를 이어붙여 저장한다."""
    with open(coco_json_path, encoding="utf-8") as f:
        coco = json.load(f)

    next_img_id = max((img["id"] for img in coco["images"]), default=0) + 1
    next_ann_id = max((ann["id"] for ann in coco["annotations"]), default=0) + 1

    existing_cat_ids = {c["id"] for c in coco["categories"]}
    existing_file_names = {img["file_name"] for img in coco["images"]}

    added_images, added_annotations, skipped_duplicates = 0, 0, 0

    os.makedirs(images_dst_dir, exist_ok=True)

    for fname, entry in new_merged.items():
        if fname in existing_file_names:
            skipped_duplicates += 1
            continue

        img_info = dict(entry["image_info"])
        img_info["id"] = next_img_id
        coco["images"].append(img_info)
        added_images += 1

        for ann in entry["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = next_img_id
            coco["annotations"].append(new_ann)
            next_ann_id += 1
            added_annotations += 1

        next_img_id += 1

        # 이미지 파일 복사 (재귀 검색으로 찾은 실제 경로 사용)
        dst = os.path.join(images_dst_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(entry["image_path"], dst)

    for dl_idx, name in matched_names.items():
        if dl_idx not in existing_cat_ids:
            coco["categories"].append({"supercategory": "pill", "id": dl_idx, "name": name})
            existing_cat_ids.add(dl_idx)

    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"  {coco_json_path}: 이미지 +{added_images}, annotation +{added_annotations}, 중복 건너뜀 {skipped_duplicates}")
    return coco


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h


def regenerate_yolo_labels(coco_train: dict, labels_train_dir: str, coco_val: dict = None, labels_val_dir: str = None):
    """train.json(전체 클래스 기준)을 기준으로 yolo 라벨(txt)과 classes.txt/data.yaml을 다시 생성한다.

    val.json/labels_val_dir도 함께 넘기면, val 라벨도 "train과 동일한 클래스 인덱스 순서"로
    다시 생성한다. train만 클래스가 늘어나고 val 라벨의 인덱스가 예전 그대로 남으면,
    같은 인덱스가 서로 다른 클래스를 가리키게 되어 검증 결과가 완전히 틀어지므로 필수 작업이다.
    """
    all_cat_ids = sorted({c["id"] for c in coco_train["categories"]})
    cat_id_to_yolo_idx = {cid: i for i, cid in enumerate(all_cat_ids)}
    cat_id_to_name = {c["id"]: c["name"] for c in coco_train["categories"]}

    def write_labels_for(coco_data: dict, out_dir: str, label: str):
        anns_by_image = defaultdict(list)
        for ann in coco_data["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        os.makedirs(out_dir, exist_ok=True)
        for img in coco_data["images"]:
            img_w, img_h = img["width"], img["height"]
            lines = []
            for ann in anns_by_image.get(img["id"], []):
                bbox = ann.get("bbox")
                if bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
                    print(f"  [경고] 잘못된 bbox 발견, 건너뜀: image_id={img['id']}, bbox={bbox}")
                    continue
                cid = ann["category_id"]
                if cid not in cat_id_to_yolo_idx:
                    print(f"  [경고] {label}에만 있는 category_id={cid} 발견 (train에 없음), 건너뜀")
                    continue
                yolo_idx = cat_id_to_yolo_idx[cid]
                xc, yc, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
                lines.append(f"{yolo_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            txt_name = img["file_name"].replace(".png", ".txt")
            with open(os.path.join(out_dir, txt_name), "w") as f:
                f.write("\n".join(lines))

    write_labels_for(coco_train, labels_train_dir, "train")
    if coco_val is not None and labels_val_dir is not None:
        write_labels_for(coco_val, labels_val_dir, "val")
        print(f"  val 라벨도 동일한 클래스 인덱스로 재생성 완료 ({labels_val_dir})")

    # classes_coco.txt 갱신
    with open(CLASSES_COCO_TXT, "w", encoding="utf-8") as f:
        for cid in all_cat_ids:
            f.write(f"{cid}: {cat_id_to_name[cid]}\n")

    # yolo_labels/classes.txt 갱신
    classes_txt_path = os.path.join(os.path.dirname(labels_train_dir), "classes.txt")
    with open(classes_txt_path, "w", encoding="utf-8") as f:
        for cid in all_cat_ids:
            f.write(f"{cat_id_to_name[cid]}\n")

    # data.yaml의 nc/names 갱신 (train/val 경로 줄은 그대로 유지)
    names_list = [cat_id_to_name[cid] for cid in all_cat_ids]
    with open(DATA_YAML, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("nc:"):
            new_lines.append(f"nc: {len(names_list)}\n")
        elif line.startswith("names:"):
            new_lines.append(f"names: {names_list}\n")
        else:
            new_lines.append(line)

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"  YOLO 라벨 재생성 완료 (클래스 수: {len(all_cat_ids)})")


def update_oversampled_list(coco_train: dict, images_train_dir: str, labels_train_dir: str, target_min: int = 9):
    """train_oversampled.txt를 다시 계산한다.

    기존 이미지는 이미 오버샘플링된 상태(같은 file_name이 여러 image_id로 존재)일 수 있으므로,
    train.json의 고유 file_name 목록을 기준으로 클래스별 등장 횟수를 다시 세고,
    target_min 미달 클래스가 포함된 이미지를 복제하여 train_oversampled.txt를 새로 만든다.
    """
    yolo_root_dir = os.path.dirname(labels_train_dir)  # data/labels
    oversampled_txt_path = os.path.join(yolo_root_dir, "train_oversampled.txt")

    # file_name 기준 고유 이미지들과, 그 이미지가 가진 category_id 집합을 구성
    anns_by_image_id = defaultdict(list)
    for ann in coco_train["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann["category_id"])

    file_names_seen = set()
    file_to_categories = {}
    for img in coco_train["images"]:
        fname = img["file_name"]
        if fname in file_names_seen:
            continue  # 이미 오버샘플링으로 중복 등록된 file_name은 한 번만 계산 대상으로 삼음
        file_names_seen.add(fname)
        file_to_categories[fname] = set(anns_by_image_id.get(img["id"], []))

    unique_file_names = list(file_to_categories.keys())

    # 클래스별 등장 이미지 수 계산 (고유 file_name 기준)
    class_img_count = defaultdict(int)
    for fname, cats in file_to_categories.items():
        for c in cats:
            class_img_count[c] += 1

    # target_min 기준으로 복제 목록 생성 (기존 오버샘플링 로직과 동일한 방식)
    oversampled = list(unique_file_names)
    for fname in unique_file_names:
        cats = file_to_categories[fname]
        if not cats:
            continue
        min_count = min(class_img_count[c] for c in cats)
        if min_count < target_min:
            repeat = int(math.ceil(target_min / min_count))
            oversampled.extend([fname] * (repeat - 1))

    with open(oversampled_txt_path, "w") as f:
        for fname in oversampled:
            f.write(f"{os.path.join(images_train_dir, fname)}\n")

    print(f"  {oversampled_txt_path} 갱신 완료: 고유 이미지 {len(unique_file_names)}개 -> 오버샘플링 후 {len(oversampled)}줄")


def delete_yolo_cache(labels_train_dir: str):
    """train.cache / val.cache를 삭제한다.

    Ultralytics는 라벨 폴더의 캐시 파일(*.cache)이 있으면, 데이터가 실제로 바뀌었어도
    다시 스캔하지 않고 예전 캐시를 그대로 재사용하는 경우가 있다. 클래스/라벨을 새로
    병합한 뒤에는 반드시 캐시를 지워야, 다음 학습이 최신 데이터를 반영한다.
    """
    labels_root = os.path.dirname(labels_train_dir)  # data/labels
    cache_paths = [
        os.path.join(labels_root, "train.cache"),
        os.path.join(labels_root, "val.cache"),
    ]
    for path in cache_paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"  캐시 삭제: {path}")
        else:
            print(f"  캐시 없음(삭제 불필요): {path}")


def main():
    print("=" * 70)
    print("기존 클래스 이름 -> id 매핑 로드 (오프셋 차이 보정용)")
    print("=" * 70)
    existing_name_to_id = load_existing_name_to_id(CLASSES_COCO_TXT)
    existing_ids = set(existing_name_to_id.values())
    print(f"  기존 클래스 {len(existing_name_to_id)}개 로드")

    print("\n" + "=" * 70)
    print(f"AI Hub 라벨 전체 병합 ({len(AIHUB_SETS)}개 세트, 이름 일치 시 기존 id로 통일)")
    print("=" * 70)
    merged_all, dl_idx_to_name = merge_all_components(AIHUB_SETS, existing_name_to_id)

    print("\n" + "=" * 70)
    if MODE == "all":
        print("모드: all — 기존과 이름이 안 겹치는 모든 새 클래스를 포함한 이미지 전부 추가")
    else:
        print(f"모드: target — 대상 클래스(dl_idx={TARGET_DL_IDX})가 포함된 이미지만 필터링")
    print("=" * 70)

    if MODE == "all":
        new_merged = filter_new_classes_only(merged_all, dl_idx_to_name, existing_ids)
    else:
        new_merged = filter_images_containing_targets(merged_all, TARGET_DL_IDX)

    if not new_merged:
        print("  [중단] 추가할 데이터를 찾지 못했습니다. 경로/설정을 확인하세요.")
        return

    print(f"  새로 추가할 이미지 수: {len(new_merged)}")
    all_cats_in_new = set()
    for entry in new_merged.values():
        all_cats_in_new |= {ann["category_id"] for ann in entry["annotations"]}
    new_class_ids = all_cats_in_new - existing_ids
    print(f"  이 이미지들에 포함된 전체 클래스 수(기존 동반 성분 포함): {len(all_cats_in_new)}")
    print(f"  이 중 완전히 새로운 클래스 수: {len(new_class_ids)}")

    relevant_names = {cid: dl_idx_to_name.get(cid, "") for cid in all_cats_in_new}

    print("\n" + "=" * 70)
    print("train.json에 병합")
    print("=" * 70)
    coco_train = append_to_coco_json(COCO_TRAIN_JSON, new_merged, relevant_names, None, IMAGES_TRAIN_DIR)

    print("\n" + "=" * 70)
    print("YOLO 라벨 재생성 (train + val 동일 클래스 인덱스로)")
    print("=" * 70)
    with open(COCO_VAL_JSON, encoding="utf-8") as f:
        coco_val = json.load(f)
    labels_val_dir = os.path.join(os.path.dirname(LABELS_TRAIN_DIR), "val")
    regenerate_yolo_labels(coco_train, LABELS_TRAIN_DIR, coco_val=coco_val, labels_val_dir=labels_val_dir)

    print("\n" + "=" * 70)
    print("오버샘플링 목록(train_oversampled.txt) 갱신")
    print("=" * 70)
    update_oversampled_list(coco_train, IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR, target_min=TARGET_MIN)

    print("\n" + "=" * 70)
    print("YOLO 캐시 삭제 (다음 학습이 최신 라벨을 반영하도록)")
    print("=" * 70)
    delete_yolo_cache(LABELS_TRAIN_DIR)

    print("\n완료. val.json은 건드리지 않았습니다 (기존 검증 기준 유지).")


if __name__ == "__main__":
    main()