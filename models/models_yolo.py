# YOLO 모델 정의
"""
YOLO 학습/추론 스크립트.
 
main.py에서 `python main.py --model yolo`로 실행하면
이 파일의 run(project_root) 함수가 호출됩니다.
 
run() 함수 안에 실제 학습/추론 코드를 구현.
 
사용 가능한 데이터 (전처리 완료 후 기준):
    project_root/data/yolo_labels/data.yaml            (학습 설정 파일, train/val 경로 및 클래스 포함)
    project_root/data/yolo_labels/train_oversampled.txt (오버샘플링 반영된 학습 이미지 목록)
    project_root/data/labels/train/*.txt               (train 라벨, 이미지 1장당 1개)
    project_root/data/labels/val/*.txt                 (val 라벨)
    project_root/data/yolo_labels/classes.txt           (0부터 시작하는 YOLO index 기준 이름 매핑)
    project_root/data/images/train, val, test           (이미지 원본)
"""
 
import json
import os
import sys
from pathlib import Path

# 이 파일을 "python models/models_yolo.py"처럼 직접 실행해도
# project_root(models 폴더의 상위 폴더)를 항상 찾을 수 있도록 경로를 보정한다.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

BASELINE_AUGMENTATION = {
    "degrees": 10.0,
    "hsv_h": 0.02,
    "hsv_s": 0.6,
    "hsv_v": 0.5,
    "fliplr": 0.3,
    "flipud": 0.0,
    "translate": 0.1,
    "scale": 0.3,
    "mosaic": 0.5,
}


def _resolve_weights(project_root: Path, weights: str) -> str:
    """프로젝트 상대 경로의 가중치가 있으면 사용하고, 없으면 Ultralytics 이름을 반환한다."""
    candidate = Path(weights)
    if candidate.is_absolute():
        return str(candidate) if candidate.exists() else weights

    local_candidates = [project_root / candidate, project_root / "weights" / candidate.name]
    local_weights = next((path for path in local_candidates if path.exists()), None)
    return str(local_weights) if local_weights else weights


def _check_training_inputs(data_yaml: Path, images_test: Path, labels_root: Path) -> None:
    missing = [str(path) for path in (data_yaml, images_test, labels_root / "train", labels_root / "val")
               if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "YOLO 입력 파일이 없습니다. 먼저 전처리를 실행하세요:\n- " + "\n- ".join(missing)
        )

    train_labels = list((labels_root / "train").glob("*.txt"))
    val_labels = list((labels_root / "val").glob("*.txt"))
    if not train_labels or not val_labels:
        raise RuntimeError("YOLO train/val 라벨이 비어 있습니다. 전처리 결과를 확인하세요.")


def _write_run_summary(save_dir: Path, config: dict, metrics) -> Path:
    """실험 조건과 validation 지표를 JSON으로 남긴다."""
    metric_values = {
        key: float(value)
        for key, value in getattr(metrics, "results_dict", {}).items()
        if isinstance(value, (int, float)) or hasattr(value, "item")
    }
    summary = {"config": config, "metrics": metric_values}
    path = save_dir / "experiment_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run(
    project_root: str,
    *,
    weights: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 4,
    device: str | None = None,
    workers: int = 0,
    run_name: str = "yolov8n_baseline",
    smoke_test: bool = False,
    predict_test: bool = False,
    conf: float = 0.001,
    iou: float = 0.7,
    degrees: float = BASELINE_AUGMENTATION["degrees"],
    hsv_h: float = BASELINE_AUGMENTATION["hsv_h"],
    hsv_s: float = BASELINE_AUGMENTATION["hsv_s"],
    hsv_v: float = BASELINE_AUGMENTATION["hsv_v"],
    fliplr: float = BASELINE_AUGMENTATION["fliplr"],
    flipud: float = BASELINE_AUGMENTATION["flipud"],
    translate: float = BASELINE_AUGMENTATION["translate"],
    scale: float = BASELINE_AUGMENTATION["scale"],
    mosaic: float = BASELINE_AUGMENTATION["mosaic"],
) -> dict:
    """YOLOv8n을 학습하고 best.pt로 validation을 평가한다.

    smoke_test=True이면 전체 연결 확인을 위해 데이터 일부와 작은 이미지 크기로
    1 epoch만 실행한다. predict_test=True이면 test 이미지의 YOLO 형식 예측 라벨도 저장한다.
    """
    root = Path(project_root).resolve()
    data_yaml = root / "data" / "yolo_labels" / "data.yaml"
    images_test = root / "data" / "images" / "test"
    labels_root = root / "data" / "labels"
    runs_root = root / "runs" / "yolo"

    _check_training_inputs(data_yaml, images_test, labels_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    # 사용자 홈의 쓰기 권한에 의존하지 않도록 프로젝트의 Git 제외 폴더를 사용한다.
    runtime_root = root / "local-review"
    yolo_config_root = runtime_root / "ultralytics"
    matplotlib_config_root = runtime_root / "matplotlib"
    yolo_config_root.mkdir(parents=True, exist_ok=True)
    matplotlib_config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_root))

    from ultralytics import YOLO

    if smoke_test:
        epochs = 1
        imgsz = min(imgsz, 320)
        batch = min(batch, 2)
        run_name = f"{run_name}_smoke"

    selected_device = None if device in (None, "", "auto") else device
    selected_weights = _resolve_weights(root, weights)
    augmentation = {
        "degrees": degrees,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "fliplr": fliplr,
        "flipud": flipud,
        "translate": translate,
        "scale": scale,
        "mosaic": mosaic,
    }
    config = {
        "weights": selected_weights,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": selected_device or "auto",
        "workers": workers,
        "run_name": run_name,
        "smoke_test": smoke_test,
        "predict_test": predict_test,
        "conf": conf,
        "iou": iou,
        "augmentation": augmentation,
    }

    print("[YOLO] 학습 시작")
    print(json.dumps(config, ensure_ascii=False, indent=2))

    model = YOLO(selected_weights)
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "seed": 42,
        "deterministic": True,
        "project": str(runs_root),
        "name": run_name,
        # 매 epoch마다 validation 성능을 계산해야 best.pt가
        # 가장 좋은 검증 성능의 가중치로 선택된다.
        "val": True,
        **augmentation,
    }
    if selected_device is not None:
        train_args["device"] = selected_device
    if smoke_test:
        train_args.update({"fraction": 0.05, "plots": False})

    model.train(**train_args)
    train_dir = Path(model.trainer.save_dir)
    best_path = train_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"학습은 끝났지만 best.pt를 찾을 수 없습니다: {best_path}")

    print(f"[YOLO] validation 평가: {best_path}")
    best_model = YOLO(str(best_path))
    val_args = {
        "data": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "project": str(runs_root),
        "name": f"{run_name}_val",
        "plots": not smoke_test,
    }
    if selected_device is not None:
        val_args["device"] = selected_device
    metrics = best_model.val(**val_args)
    summary_path = _write_run_summary(train_dir, config, metrics)

    prediction_count = 0
    prediction_dir = None
    if predict_test:
        print(f"[YOLO] test 추론 시작: {images_test}")
        predict_args = {
            "source": str(images_test),
            "imgsz": imgsz,
            "project": str(runs_root),
            "name": f"{run_name}_predict",
            "save": False,
            "save_txt": True,
            "save_conf": True,
            "stream": True,
            "conf": conf,
            "iou": iou,
        }
        if selected_device is not None:
            predict_args["device"] = selected_device
        for _ in best_model.predict(**predict_args):
            prediction_count += 1
        prediction_dir = Path(best_model.predictor.save_dir)
        print(f"[YOLO] test 추론 {prediction_count}장 완료: {prediction_dir}")

    print(f"[YOLO] 완료: best.pt={best_path}")
    print(f"[YOLO] 실험 요약={summary_path}")
    return {
        "best_weights": str(best_path),
        "summary": str(summary_path),
        "prediction_images": prediction_count,
        "predictions": str(prediction_dir) if prediction_dir else None,
    }
 
if __name__ == "__main__":
    # 단독 실행 테스트용 (project_root를 이 파일 기준 상위 폴더로 가정)
    run(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
