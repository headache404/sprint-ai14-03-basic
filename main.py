"""
경구약제 객체 검출 프로젝트 - 실행 진입점

전처리 옵션:
    python main.py                 # 전처리 산출물이 없으면 자동 실행, 있으면 다시 할지 물어봄
    python main.py --preprocess    # 무조건 전처리를 새로 실행 (기존 산출물 덮어씀)
    python main.py --skip          # 전처리를 건너뛰고 기존 산출물 그대로 사용

모델 실행 옵션 (전처리 이후에 실행됨):
    python main.py --model none    # 모델 실행 안 함 (기본값, 전처리만 수행)
    python main.py --model faster  # Faster R-CNN만 실행 (models/models_faster.py의 run() 호출)
    python main.py --model yolo    # YOLO만 실행 (models/models_yolo.py의 run() 호출)
    python main.py --model both    # 둘 다 순서대로 실행

모델 단독 실행 (전처리 후):
    python .\models\models_faster.py
    python .\models\models_yolo.py

조합 예시:
    python main.py --skip --model yolo         # 전처리는 건너뛰고 YOLO만 실행
    python main.py --preprocess --model both   # 전처리부터 새로 하고 두 모델 다 실행
"""

import os
import argparse
import importlib

from utils import utils

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 전처리 완료 여부를 판단할 기준 파일 (이 파일들이 있으면 "이미 전처리된 상태"로 간주)
DONE_MARKERS = [
    os.path.join(PROJECT_ROOT, "data", "coco_annotations", "train.json"),
    os.path.join(PROJECT_ROOT, "data", "coco_annotations", "val.json"),
    os.path.join(PROJECT_ROOT, "data", "yolo_labels", "data.yaml"),
]

# --model 옵션 값 -> (표시용 이름, import할 모듈 경로)
MODEL_MODULES = {
    "faster": ("Faster R-CNN", "models.models_faster"),
    "yolo": ("YOLO", "models.models_yolo"),
}


def already_preprocessed() -> bool:
    return all(os.path.exists(p) for p in DONE_MARKERS)


def ask_user_run_preprocessing() -> bool:
    """터미널에서 팀원에게 전처리를 다시 할지 물어본다."""
    while True:
        answer = input(
            "기존 전처리 산출물(train.json/val.json/data.yaml)이 이미 존재합니다.\n"
            "전처리를 다시 실행할까요? [y/N]: "
        ).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("y 또는 n으로 답해주세요.")


def run_preprocessing_step(args) -> None:
    """--preprocess / --skip 옵션에 따라 전처리 실행 여부를 결정하고 수행한다."""
    if args.skip:
        do_run = False
    elif args.preprocess:
        do_run = True
    else:
        # 옵션을 안 준 경우: 산출물이 없으면 자동 실행, 있으면 물어봄
        do_run = (not already_preprocessed()) or ask_user_run_preprocessing()

    if not do_run:
        print("전처리를 건너뜁니다. 기존 data/ 산출물을 그대로 사용합니다.")
        return

    utils.run_pipeline(
        project_root=PROJECT_ROOT,
        target_min=args.target_min,
        val_ratio=args.val_ratio,
    )


def run_selected_models(model_choice: str) -> None:
    """--model 옵션에 따라 지정된 모델 스크립트의 run(project_root)를 호출한다."""
    if model_choice == "none":
        return

    selected = ["faster", "yolo"] if model_choice == "both" else [model_choice]

    for key in selected:
        label, module_name = MODEL_MODULES[key]
        print("\n" + "=" * 60)
        print(f"{label} 모델 실행")
        print("=" * 60)

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"[오류] {module_name} 모듈을 불러올 수 없습니다: {e}")
            continue

        run_func = getattr(module, "run", None)
        if run_func is None:
            print(f"[안내] {module_name}.py에 run(project_root) 함수가 아직 구현되지 않았습니다.")
            continue

        try:
            run_func(PROJECT_ROOT)
        except NotImplementedError as e:
            print(f"[안내] {label} 담당자 구현 대기 중: {e}")


def main():
    parser = argparse.ArgumentParser(description="경구약제 데이터 전처리 + 모델 실행")

    prep_group = parser.add_mutually_exclusive_group()
    prep_group.add_argument(
        "--preprocess", action="store_true",
        help="기존 산출물 여부와 상관없이 전처리를 강제로 다시 실행"
    )
    prep_group.add_argument(
        "--skip", action="store_true",
        help="전처리를 건너뛰고 기존 산출물을 그대로 사용"
    )
    parser.add_argument(
        "--target-min", type=int, default=9,
        help="오버샘플링 시 클래스별 최소 등장 횟수 목표값 (기본 9)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="validation 데이터 비율 (기본 0.2)"
    )
    parser.add_argument(
        "--model", choices=["none", "faster", "yolo", "both"], default="none",
        help="전처리 이후 실행할 모델 선택 (기본값: none = 실행 안 함)"
    )
    args = parser.parse_args()

    run_preprocessing_step(args)
    run_selected_models(args.model)


if __name__ == "__main__":
    main()