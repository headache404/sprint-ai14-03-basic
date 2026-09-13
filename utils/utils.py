"""
경구약제 객체 검출 프로젝트 - 전처리 유틸 모듈

이 파일은 Colab 노트북에서 검증한 EDA/전처리 로직을
1) JSON 병합
2) bbox 유효성 필터링 및 이상치 제거
3) ID 재부여
4) train/val split (iterative stratification + 희귀 클래스 보정)
5) 오버샘플링
6) COCO JSON 저장 (Faster R-CNN용)
7) YOLO 라벨(txt) 저장 (YOLO용)
8) 이미지 파일 복사
위 순서로 진행되며 main.py에서 run_pipeline()을 호출해서 사용합니다.
"""

import os
import json
import glob
import shutil
from collections import defaultdict, Counter

import numpy as np

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    MultilabelStratifiedShuffleSplit = None


# ---------------------------------------------------------------------------
# 1. JSON 병합 (file_name 기준으로 여러 JSON을 하나로 묶기)
# ---------------------------------------------------------------------------
def merge_annotations(train_images_dir: str, train_annotations_dir: str) -> dict:
    """같은 file_name을 가리키는 여러 JSON의 annotation을 한 이미지 기준으로 병합한다."""
    json_files = glob.glob(os.path.join(train_annotations_dir, "**", "*.json"), recursive=True)

    merged = defaultdict(lambda: {"image_info": None, "annotations": [], "categories": {}})

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)

        img_info = d["images"][0]
        fname = img_info["file_name"]

        entry = merged[fname]
        if entry["image_info"] is None:
            entry["image_info"] = img_info

        entry["annotations"].extend(d["annotations"])
        for cat in d["categories"]:
            entry["categories"][cat["id"]] = cat

    print(f"[병합] JSON {len(json_files)}개 -> 이미지 {len(merged)}개로 통합")
    return merged


def validate_merge(merged: dict, train_images_dir: str) -> None:
    """이미지 폴더와 병합 결과가 완전히 매칭되는지 확인한다."""
    image_names = {os.path.basename(p) for p in glob.glob(os.path.join(train_images_dir, "*.png"))}
    merged_names = set(merged.keys())

    missing_json = image_names - merged_names
    missing_image = merged_names - image_names

    print(f"[검증] 이미지엔 있는데 JSON 없는 경우: {len(missing_json)}개")
    print(f"[검증] JSON엔 있는데 이미지 없는 경우: {len(missing_image)}개")

    if missing_json or missing_image:
        raise ValueError("이미지-JSON 매칭에 불일치가 발견되었습니다. 원본 데이터를 다시 확인하세요.")


# ---------------------------------------------------------------------------
# 2. 병합 결과를 하나의 리스트(final_images / final_annotations)로 조립 + ID 재부여
# ---------------------------------------------------------------------------
def build_final_lists(merged: dict):
    """병합된 dict를 image_id가 부여된 images/annotations 리스트로 변환한다."""
    final_images = []
    final_annotations = []
    final_categories = {}

    for new_img_id, (fname, entry) in enumerate(merged.items(), start=1):
        img_info = dict(entry["image_info"])
        img_info["id"] = new_img_id
        final_images.append(img_info)

        for ann in entry["annotations"]:
            ann = dict(ann)
            ann["image_id"] = new_img_id
            final_annotations.append(ann)

        final_categories.update(entry["categories"])

    print(f"[조립] 이미지 {len(final_images)}개 / annotation {len(final_annotations)}개 / 클래스 {len(final_categories)}개")
    return final_images, final_annotations, final_categories


# ---------------------------------------------------------------------------
# 3. bbox 유효성 필터링 + 이상치(범위 초과) 제거 + annotation_id 재부여
# ---------------------------------------------------------------------------
def filter_and_clean_bbox(final_images: list, final_annotations: list) -> list:
    """bbox 존재/유효성(길이 4) 필터링 후, 이미지 범위를 벗어난 bbox를 제거한다."""
    # 존재 + 유효성 필터링
    valid_annotations = [
        ann for ann in final_annotations
        if ann.get("bbox") not in (None, []) and isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4
    ]
    removed_invalid = len(final_annotations) - len(valid_annotations)
    print(f"[bbox 필터링] 무효 bbox 제거: {removed_invalid}개 (남은 개수: {len(valid_annotations)})")

    # 이미지 범위를 벗어난 bbox 제거 (out_of_image_bounds)
    image_size_by_id = {img["id"]: (img["width"], img["height"]) for img in final_images}

    def in_bounds(ann):
        w_img, h_img = image_size_by_id[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        return x >= 0 and y >= 0 and (x + w) <= w_img and (y + h) <= h_img

    before = len(valid_annotations)
    valid_annotations = [ann for ann in valid_annotations if in_bounds(ann)]
    print(f"[bbox 이상치 제거] 범위 초과 제거: {before - len(valid_annotations)}개 (남은 개수: {len(valid_annotations)})")

    # annotation_id 재부여
    for new_ann_id, ann in enumerate(valid_annotations, start=1):
        ann["id"] = new_ann_id

    return valid_annotations


# ---------------------------------------------------------------------------
# 4. train/val split (multilabel iterative stratification + 희귀 클래스 강제 배정)
# ---------------------------------------------------------------------------
def split_train_val(merged: dict, val_ratio: float = 0.2, random_state: int = 42):
    """클래스 불균형을 고려한 stratified split. 모든 클래스가 train/val 양쪽에 존재하도록 보정한다."""
    if MultilabelStratifiedShuffleSplit is None:
        raise ImportError("iterative-stratification 패키지가 필요합니다: pip install iterative-stratification")

    image_names = list(merged.keys())
    all_cat_ids = sorted({ann["category_id"] for e in merged.values() for ann in e["annotations"]})
    cat_id_to_idx = {cid: i for i, cid in enumerate(all_cat_ids)}

    Y = np.zeros((len(image_names), len(all_cat_ids)), dtype=int)
    for i, fname in enumerate(image_names):
        for ann in merged[fname]["annotations"]:
            Y[i, cat_id_to_idx[ann["category_id"]]] = 1
    X = np.array(image_names).reshape(-1, 1)

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(msss.split(X, Y))
    train_files = [image_names[i] for i in train_idx]
    val_files = [image_names[i] for i in val_idx]

    # 희귀 클래스(양쪽 중 한쪽에만 존재하는 클래스) 보정
    def classes_in(files):
        return {ann["category_id"] for f in files for ann in merged[f]["annotations"]}

    missing_in_val = set(all_cat_ids) - classes_in(val_files)
    if missing_in_val:
        images_by_rare_class = defaultdict(list)
        for fname in image_names:
            cats = {ann["category_id"] for ann in merged[fname]["annotations"]}
            for c in cats & missing_in_val:
                images_by_rare_class[c].append(fname)

        forced_val, forced_train = set(), set()
        for cat, files in images_by_rare_class.items():
            forced_val.add(files[0])
            forced_train.update(files[1:])

        train_files = [f for f in train_files if f not in forced_val] + list(forced_train)
        val_files = [f for f in val_files if f not in forced_train] + list(forced_val)
        # 중복 제거 (혹시 겹치면)
        train_files = list(dict.fromkeys(train_files))
        val_files = list(dict.fromkeys(val_files))

    missing_train_final = set(all_cat_ids) - classes_in(train_files)
    missing_val_final = set(all_cat_ids) - classes_in(val_files)
    print(f"[split] train {len(train_files)}장 / val {len(val_files)}장")
    print(f"[split] train에 없는 클래스: {missing_train_final}")
    print(f"[split] val에 없는 클래스: {missing_val_final}")

    return train_files, val_files


# ---------------------------------------------------------------------------
# 5. 오버샘플링 (train에만 적용)
# ---------------------------------------------------------------------------
def oversample_train(train_files: list, merged: dict, target_min: int = 9) -> list:
    """클래스별 최소 등장 횟수가 target_min 이상이 되도록 희귀 클래스가 포함된 이미지를 복제한다."""
    class_img_count = Counter()
    for fname in train_files:
        cats = {ann["category_id"] for ann in merged[fname]["annotations"]}
        for c in cats:
            class_img_count[c] += 1

    oversampled = list(train_files)
    for fname in train_files:
        cats = {ann["category_id"] for ann in merged[fname]["annotations"]}
        min_count = min(class_img_count[c] for c in cats)
        if min_count < target_min:
            repeat = int(np.ceil(target_min / min_count))
            oversampled.extend([fname] * (repeat - 1))

    new_counts = Counter()
    for fname in oversampled:
        cats = {ann["category_id"] for ann in merged[fname]["annotations"]}
        for c in cats:
            new_counts[c] += 1

    print(f"[오버샘플링] train {len(train_files)}장 -> {len(oversampled)}개 (목표 최소 등장 {target_min}, 실제 최소 {min(new_counts.values())})")
    return oversampled


# ---------------------------------------------------------------------------
# 6. COCO JSON 조립 및 저장
# ---------------------------------------------------------------------------
def build_coco_split(file_list: list, merged: dict, valid_annotations: list, final_images: list, final_categories: dict):
    """파일명 리스트(오버샘플링 반영 가능)를 받아 새 image_id/annotation_id를 부여한 COCO dict를 만든다."""
    fname_to_orig_id = {img["file_name"]: img["id"] for img in final_images}
    anns_by_orig_id = defaultdict(list)
    for ann in valid_annotations:
        anns_by_orig_id[ann["image_id"]].append(ann)

    images_out, annotations_out = [], []
    img_id_counter, ann_id_counter = 1, 1

    for fname in file_list:
        entry = merged[fname]
        img_info = dict(entry["image_info"])
        img_info["id"] = img_id_counter
        images_out.append(img_info)

        orig_id = fname_to_orig_id[fname]
        for ann in anns_by_orig_id.get(orig_id, []):
            new_ann = dict(ann)
            new_ann["id"] = ann_id_counter
            new_ann["image_id"] = img_id_counter
            annotations_out.append(new_ann)
            ann_id_counter += 1

        img_id_counter += 1

    coco_dict = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": list(final_categories.values()),
        "type": "instances",
    }
    return coco_dict


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[저장] {path} (이미지 {len(data['images'])} / annotation {len(data['annotations'])})")


# ---------------------------------------------------------------------------
# 7. 이미지 파일 복사 (train/val/test)
# ---------------------------------------------------------------------------
def copy_images(file_list, src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for fname in set(file_list):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    print(f"[이미지 복사] {dst_dir} -> {len(os.listdir(dst_dir))}개")


def copy_test_images(test_images_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for f in glob.glob(os.path.join(test_images_dir, "*.png")):
        dst = os.path.join(dst_dir, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy(f, dst)
    print(f"[이미지 복사] {dst_dir} -> {len(os.listdir(dst_dir))}개")


# ---------------------------------------------------------------------------
# 8. YOLO 라벨(txt) 생성
# ---------------------------------------------------------------------------
def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h


def write_yolo_labels(file_list, merged: dict, valid_annotations: list, final_images: list,
                       cat_id_to_yolo_idx: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fname_to_orig_id = {img["file_name"]: img["id"] for img in final_images}
    anns_by_orig_id = defaultdict(list)
    for ann in valid_annotations:
        anns_by_orig_id[ann["image_id"]].append(ann)

    for fname in set(file_list):
        entry = merged[fname]
        img_w, img_h = entry["image_info"]["width"], entry["image_info"]["height"]
        orig_id = fname_to_orig_id[fname]

        lines = []
        for ann in anns_by_orig_id.get(orig_id, []):
            yolo_idx = cat_id_to_yolo_idx[ann["category_id"]]
            xc, yc, w, h = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            lines.append(f"{yolo_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        txt_name = fname.replace(".png", ".txt")
        with open(os.path.join(out_dir, txt_name), "w") as f:
            f.write("\n".join(lines))

    print(f"[YOLO 라벨] {out_dir} -> {len(os.listdir(out_dir))}개")


def write_yolo_support_files(oversampled_train_files: list, all_cat_ids: list, cat_id_to_name: dict,
                              yolo_root_dir: str, images_train_rel: str = "../images/train",
                              images_val_rel: str = "../images/val") -> None:
    """train_oversampled.txt, data.yaml, classes.txt를 생성한다."""
    os.makedirs(yolo_root_dir, exist_ok=True)

    # 오버샘플링 반영 train 이미지 경로 목록
    oversampled_txt_path = os.path.join(yolo_root_dir, "train_oversampled.txt")
    with open(oversampled_txt_path, "w") as f:
        for fname in oversampled_train_files:
            f.write(f"{images_train_rel}/{fname}\n")
    print(f"[YOLO 목록] {oversampled_txt_path} -> {len(oversampled_train_files)}줄")

    # data.yaml
    names_list = [cat_id_to_name[cid] for cid in all_cat_ids]
    yaml_content = (
        f"train: ./train_oversampled.txt\n"
        f"val: {images_val_rel}\n\n"
        f"nc: {len(names_list)}\n"
        f"names: {names_list}\n"
    )
    with open(os.path.join(yolo_root_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # classes.txt (index 순서 = names_list 순서)
    with open(os.path.join(yolo_root_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for name in names_list:
            f.write(f"{name}\n")

    print(f"[YOLO 설정] data.yaml / classes.txt 생성 완료 ({yolo_root_dir})")


# ---------------------------------------------------------------------------
# 0. 기존 산출물 초기화 (재실행 시 이전 split의 찌꺼기 파일이 남지 않도록)
# ---------------------------------------------------------------------------
def clean_output_dirs(project_root: str) -> None:
    """재실행 전, 이미지/라벨이 쌓이는 폴더를 완전히 비운다.

    JSON/yaml/txt 파일은 매번 새로 덮어쓰기 때문에 문제가 없지만,
    이미지·라벨 폴더는 "이미 있으면 건너뛰기" 방식이라 split이 바뀌면
    예전 split 기준의 파일이 새 폴더에 그대로 남을 수 있다. 이를 방지하기 위해
    파이프라인 시작 시 관련 폴더를 통째로 삭제 후 재생성한다.
    """
    dirs_to_clean = [
        os.path.join(project_root, "data", "images", "train"),
        os.path.join(project_root, "data", "images", "val"),
        os.path.join(project_root, "data", "images", "test"),
        os.path.join(project_root, "data", "yolo_labels", "train"),
        os.path.join(project_root, "data", "yolo_labels", "val"),
    ]
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"[초기화] {d} 삭제됨")
        os.makedirs(d, exist_ok=True)


def write_coco_classes_txt(final_categories: dict, out_path: str) -> None:
    """category_id: 이름 형식의 참고용 매핑 파일을 생성한다 (COCO category_id 그대로 사용)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for cid in sorted(final_categories.keys()):
            f.write(f"{cid}: {final_categories[cid]['name']}\n")
    print(f"[COCO 클래스 매핑] {out_path} 생성 완료 ({len(final_categories)}개)")


# ---------------------------------------------------------------------------
# 전체 파이프라인 실행
# ---------------------------------------------------------------------------
def run_pipeline(
    project_root: str,
    raw_data_dirname: str = "sprint_ai_project1_data",
    target_min: int = 9,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> None:
    """전처리 전체 과정을 순서대로 실행한다."""
    raw_dir = os.path.join(project_root, raw_data_dirname)
    train_images_dir = os.path.join(raw_dir, "train_images")
    train_annotations_dir = os.path.join(raw_dir, "train_annotations")
    test_images_dir = os.path.join(raw_dir, "test_images")

    coco_out_dir = os.path.join(project_root, "data", "coco_annotations")
    images_out_dir = os.path.join(project_root, "data", "images")
    yolo_root_dir = os.path.join(project_root, "data", "yolo_labels")
    yolo_labels_dir = yolo_root_dir

    print("=" * 60)
    print("0. 기존 산출물 초기화")
    print("=" * 60)
    clean_output_dirs(project_root)

    print("\n" + "=" * 60)
    print("1~2. JSON 병합 및 검증")
    print("=" * 60)
    merged = merge_annotations(train_images_dir, train_annotations_dir)
    validate_merge(merged, train_images_dir)
    final_images, final_annotations, final_categories = build_final_lists(merged)
    cat_id_to_name = {cid: cat["name"] for cid, cat in final_categories.items()}

    print("\n" + "=" * 60)
    print("3. bbox 필터링 및 이상치 제거")
    print("=" * 60)
    valid_annotations = filter_and_clean_bbox(final_images, final_annotations)

    print("\n" + "=" * 60)
    print("4. train/val split")
    print("=" * 60)
    train_files, val_files = split_train_val(merged, val_ratio=val_ratio, random_state=random_state)

    print("\n" + "=" * 60)
    print("5. 오버샘플링 (train만 적용)")
    print("=" * 60)
    oversampled_train_files = oversample_train(train_files, merged, target_min=target_min)

    print("\n" + "=" * 60)
    print("6. COCO JSON 저장 (Faster R-CNN용)")
    print("=" * 60)
    train_coco = build_coco_split(oversampled_train_files, merged, valid_annotations, final_images, final_categories)
    val_coco = build_coco_split(val_files, merged, valid_annotations, final_images, final_categories)
    save_json(train_coco, os.path.join(coco_out_dir, "train.json"))
    save_json(val_coco, os.path.join(coco_out_dir, "val.json"))
    write_coco_classes_txt(final_categories, os.path.join(project_root, "data", "classes_coco.txt"))

    print("\n" + "=" * 60)
    print("7. 이미지 파일 복사")
    print("=" * 60)
    copy_images(train_files, train_images_dir, os.path.join(images_out_dir, "train"))
    copy_images(val_files, train_images_dir, os.path.join(images_out_dir, "val"))
    copy_test_images(test_images_dir, os.path.join(images_out_dir, "test"))

    print("\n" + "=" * 60)
    print("8. YOLO 라벨 및 설정 파일 생성")
    print("=" * 60)
    all_cat_ids = sorted(final_categories.keys())
    cat_id_to_yolo_idx = {cid: i for i, cid in enumerate(all_cat_ids)}

    write_yolo_labels(train_files, merged, valid_annotations, final_images, cat_id_to_yolo_idx,
                       os.path.join(yolo_labels_dir, "train"))
    write_yolo_labels(val_files, merged, valid_annotations, final_images, cat_id_to_yolo_idx,
                       os.path.join(yolo_labels_dir, "val"))
    write_yolo_support_files(oversampled_train_files, all_cat_ids, cat_id_to_name,
                              yolo_root_dir=yolo_root_dir)

    print("\n" + "=" * 60)
    print("전처리 파이프라인 완료")
    print("=" * 60)