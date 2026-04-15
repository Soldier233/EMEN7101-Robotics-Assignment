import argparse
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

from bow_retrieval import IMAGE_EXTENSIONS, bow_retrieval_pipeline, load_image_paths


OXFORD_GT_SUFFIXES = ("_query.txt", "_good.txt", "_ok.txt", "_junk.txt")


def _draw_scene(scene_id: int, variant: int, is_query: bool, size: int = 320) -> np.ndarray:
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    scene_words = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPS"]
    color_map = [
        ((20, 40), (300, 60), (10, 10, 10)),
        ((30, 260), (290, 280), (20, 20, 20)),
        ((60, 80), (260, 240), (0, 0, 0)),
        ((160, 20), (180, 300), (0, 0, 0)),
    ]

    for start, end, color in color_map:
        thickness = 2 + (scene_id % 3)
        cv2.line(image, start, end, color, thickness)

    anchor_sets = {
        0: [(38, 38), (280, 44), (48, 278), (250, 250), (155, 155)],
        1: [(60, 60), (260, 65), (88, 250), (220, 235), (160, 120)],
        2: [(40, 90), (278, 92), (65, 225), (255, 225), (160, 270)],
        3: [(55, 55), (265, 55), (55, 265), (265, 265), (160, 95)],
        4: [(85, 45), (235, 45), (50, 220), (270, 220), (160, 270)],
    }
    for cx, cy in anchor_sets[scene_id]:
        cv2.rectangle(image, (cx - 6, cy - 6), (cx + 6, cy + 6), (0, 0, 0), -1)

    if scene_id == 0:
        cv2.rectangle(image, (40, 40), (140, 150), (0, 0, 0), 3)
        cv2.circle(image, (230, 100), 40, (0, 0, 0), 3)
        cv2.putText(image, "A", (110, 250), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    elif scene_id == 1:
        pts = np.array([[80, 220], [160, 70], [250, 220]], dtype=np.int32)
        cv2.polylines(image, [pts], True, (0, 0, 0), 4)
        cv2.circle(image, (90, 90), 28, (0, 0, 0), -1)
        cv2.putText(image, "B", (185, 250), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    elif scene_id == 2:
        for offset in range(40, 280, 45):
            cv2.line(image, (offset, 40), (offset, 280), (0, 0, 0), 2)
        cv2.circle(image, (160, 160), 55, (0, 0, 0), 4)
        cv2.putText(image, "C", (118, 172), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
    elif scene_id == 3:
        for row in range(70, 250, 45):
            for col in range(70, 250, 45):
                cv2.rectangle(image, (col, row), (col + 18, row + 18), (0, 0, 0), 2)
        diamond = np.array([[160, 55], [255, 160], [160, 265], [65, 160]], dtype=np.int32)
        cv2.polylines(image, [diamond], True, (0, 0, 0), 4)
        cv2.putText(image, "D", (135, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    else:
        cv2.circle(image, (160, 160), 95, (0, 0, 0), 4)
        cv2.circle(image, (160, 160), 45, (0, 0, 0), 4)
        zigzag = np.array([[45, 240], [95, 195], [145, 240], [195, 195], [245, 240]], dtype=np.int32)
        cv2.polylines(image, [zigzag], False, (0, 0, 0), 4)
        cv2.putText(image, "E", (145, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)

    cv2.putText(
        image,
        scene_words[scene_id],
        (28, 304),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        image,
        f"S{scene_id+1}",
        (230, 34),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        0.95,
        (0, 0, 0),
        1,
    )

    rng = np.random.default_rng(scene_id * 100 + variant + (50 if is_query else 0))
    for _ in range(24):
        center = rng.integers(20, size - 20, size=2)
        radius = int(rng.integers(3, 8))
        shade = int(rng.integers(40, 120))
        cv2.circle(image, tuple(center.tolist()), radius, (shade, shade, shade), -1)

    if is_query:
        angle = [-6, 4, 7, -4, 5][scene_id]
        scale = 1.0 + 0.02 * ((variant % 3) - 1)
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
        matrix[:, 2] += np.array([8 + 2 * scene_id, -6 + variant])
        image = cv2.warpAffine(image, matrix, (size, size), borderValue=(255, 255, 255))
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.7)
    else:
        dx = int((variant - 1) * 6)
        dy = int((scene_id - 2) * 2)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(image, matrix, (size, size), borderValue=(255, 255, 255))

    return image


def generate_demo_dataset(reference_dir: str, query_dir: str) -> Dict[str, int]:
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    scene_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    ref_count = 0
    query_count = 0

    for scene_id, scene_name in enumerate(scene_names):
        for variant in range(3):
            image = _draw_scene(scene_id, variant, is_query=False)
            path = os.path.join(reference_dir, f"{scene_name}_ref{variant+1}.png")
            cv2.imwrite(path, image)
            ref_count += 1

        image = _draw_scene(scene_id, 0, is_query=True)
        path = os.path.join(query_dir, f"{scene_name}_query.png")
        cv2.imwrite(path, image)
        query_count += 1

    return {"reference_images": ref_count, "query_images": query_count}


def _build_demo_bundle(reference_paths: Sequence[str], query_paths: Sequence[str]) -> Dict[str, object]:
    return {
        "mode": "demo",
        "reference_images": [
            {
                "path": path,
                "image_id": Path(path).stem,
                "label": Path(path).stem.split("_")[0],
            }
            for path in reference_paths
        ],
        "queries": [
            {
                "query_name": Path(path).stem,
                "query_path": path,
                "query_id": Path(path).stem,
                "label": Path(path).stem.split("_")[0],
                "source_image_id": "",
                "bbox": None,
                "relevant_ids": [],
                "junk_ids": [],
            }
            for path in query_paths
        ],
    }


def ensure_demo_dataset(config: Dict[str, object], force_generate: bool = False) -> Dict[str, object]:
    demo_cfg = config["dataset"]["demo"]
    reference_dir = str(demo_cfg["reference_dir"])
    query_dir = str(demo_cfg["query_dir"])
    auto_generate = bool(demo_cfg.get("auto_generate_if_missing", True))

    reference_paths = load_image_paths(reference_dir, IMAGE_EXTENSIONS)
    query_paths = load_image_paths(query_dir, IMAGE_EXTENSIONS)

    if force_generate or ((not reference_paths or not query_paths) and auto_generate):
        generate_demo_dataset(reference_dir, query_dir)
        reference_paths = load_image_paths(reference_dir, IMAGE_EXTENSIONS)
        query_paths = load_image_paths(query_dir, IMAGE_EXTENSIONS)

    if not reference_paths or not query_paths:
        raise RuntimeError("Demo dataset folders are empty and auto generation is disabled or failed.")
    return _build_demo_bundle(reference_paths, query_paths)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with urllib.request.urlopen(url) as response, open(destination, "wb") as handle:
        handle.write(response.read())


def extract_tar_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(destination)


def _flatten_single_nested_directory(directory: Path) -> None:
    if not directory.exists():
        return
    entries = list(directory.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    nested_dir = entries[0]
    moved_any = False
    for child in nested_dir.iterdir():
        target = directory / child.name
        if target.exists():
            continue
        child.rename(target)
        moved_any = True
    if moved_any:
        try:
            nested_dir.rmdir()
        except OSError:
            pass


def _collect_gt_groups(ground_truth_dir: Path) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = {}
    for path in sorted(ground_truth_dir.glob("*.txt")):
        name = path.name
        matched = False
        for suffix in OXFORD_GT_SUFFIXES:
            if name.endswith(suffix):
                query_name = name[: -len(suffix)]
                groups.setdefault(query_name, {})[suffix] = path
                matched = True
                break
        if not matched:
            continue
    return groups


def _parse_query_file(query_file: Path) -> Tuple[str, Tuple[int, int, int, int]]:
    raw = query_file.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Query file is empty: {query_file}")
    parts = raw.split()
    if len(parts) < 5:
        raise ValueError(f"Unexpected Oxford query format in {query_file}: {raw}")

    image_id = parts[0]
    if image_id.startswith("oxc1_"):
        image_id = image_id[len("oxc1_") :]

    bbox_values = [int(round(float(value))) for value in parts[1:5]]
    x1, y1, x2, y2 = bbox_values
    return image_id, (x1, y1, x2, y2)


def _read_id_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            values.append(line)
    return values


def _resolve_image_path(images_dir: Path, image_id: str) -> Path:
    direct = images_dir / f"{image_id}.jpg"
    if direct.exists():
        return direct
    matches = list(images_dir.glob(f"{image_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find Oxford image for id '{image_id}' in {images_dir}")


def _clip_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, x1 + 1), width)
    y2 = min(max(y2, y1 + 1), height)
    return x1, y1, x2, y2


def _generate_query_crop(source_image_path: Path, bbox: Tuple[int, int, int, int], output_path: Path) -> Tuple[int, int, int, int]:
    image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read source Oxford image: {source_image_path}")
    height, width = image.shape[:2]
    clipped_bbox = _clip_bbox(bbox, width, height)
    x1, y1, x2, y2 = clipped_bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Query crop is empty for {source_image_path} with bbox {bbox}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)
    return clipped_bbox


def _validate_oxford_layout(images_dir: Path, ground_truth_dir: Path) -> None:
    if not images_dir.is_dir():
        raise RuntimeError(f"Oxford images directory is missing: {images_dir}")
    if not ground_truth_dir.is_dir():
        raise RuntimeError(f"Oxford ground-truth directory is missing: {ground_truth_dir}")
    image_count = len(load_image_paths(str(images_dir), IMAGE_EXTENSIONS))
    if image_count == 0:
        raise RuntimeError(f"Oxford images directory is empty: {images_dir}")
    groups = _collect_gt_groups(ground_truth_dir)
    if not groups:
        raise RuntimeError(f"No Oxford ground-truth query files found in {ground_truth_dir}")
    for query_name, files in groups.items():
        missing = [suffix for suffix in OXFORD_GT_SUFFIXES if suffix not in files]
        if missing:
            raise RuntimeError(f"Oxford query '{query_name}' is missing GT files: {missing}")


def ensure_oxford5k_dataset(config: Dict[str, object]) -> Dict[str, object]:
    oxford_cfg = config["dataset"]["oxford5k"]
    data_root = Path(str(oxford_cfg["data_root"]))
    images_dir = Path(str(oxford_cfg["images_dir"]))
    ground_truth_dir = Path(str(oxford_cfg["ground_truth_dir"]))
    query_crops_dir = Path(str(oxford_cfg["query_crops_dir"]))
    archive_dir = Path(str(oxford_cfg["archive_dir"]))
    auto_download = bool(oxford_cfg.get("auto_download", True))
    images_url = str(oxford_cfg.get("download_images_url", "")).strip()
    gt_url = str(oxford_cfg.get("download_gt_url", "")).strip()

    data_root.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    query_crops_dir.mkdir(parents=True, exist_ok=True)

    existing_images = load_image_paths(str(images_dir), IMAGE_EXTENSIONS)
    gt_groups = _collect_gt_groups(ground_truth_dir) if ground_truth_dir.is_dir() else {}
    needs_download = not existing_images or not gt_groups

    if needs_download:
        if not auto_download:
            raise RuntimeError("Oxford5k data is missing and auto_download is disabled.")
        if not images_url or not gt_url:
            raise RuntimeError("Oxford5k download URLs are not configured.")

        images_archive = archive_dir / Path(images_url).name
        gt_archive = archive_dir / Path(gt_url).name
        download_file(images_url, images_archive)
        download_file(gt_url, gt_archive)

        if not existing_images:
            extract_tar_archive(images_archive, images_dir)
            _flatten_single_nested_directory(images_dir)
        if not gt_groups:
            extract_tar_archive(gt_archive, ground_truth_dir)
            _flatten_single_nested_directory(ground_truth_dir)

    _validate_oxford_layout(images_dir, ground_truth_dir)

    reference_paths = load_image_paths(str(images_dir), IMAGE_EXTENSIONS)
    reference_images = [
        {
            "path": path,
            "image_id": Path(path).stem,
            "label": Path(path).stem.split("_")[0],
        }
        for path in reference_paths
    ]

    queries = []
    for query_name, files in sorted(_collect_gt_groups(ground_truth_dir).items()):
        source_image_id, bbox = _parse_query_file(files["_query.txt"])
        source_image_path = _resolve_image_path(images_dir, source_image_id)
        crop_path = query_crops_dir / f"{query_name}.jpg"
        clipped_bbox = _generate_query_crop(source_image_path, bbox, crop_path)
        good_ids = set(_read_id_list(files["_good.txt"]))
        ok_ids = set(_read_id_list(files["_ok.txt"]))
        junk_ids = set(_read_id_list(files["_junk.txt"]))
        relevant_ids = sorted(good_ids | ok_ids)
        queries.append(
            {
                "query_name": query_name,
                "query_path": str(crop_path),
                "query_id": query_name,
                "label": source_image_id.split("_")[0],
                "source_image_id": source_image_id,
                "source_image_path": str(source_image_path),
                "bbox": clipped_bbox,
                "relevant_ids": relevant_ids,
                "junk_ids": sorted(junk_ids),
            }
        )

    return {
        "mode": "oxford5k",
        "reference_images": reference_images,
        "queries": queries,
        "metadata": {
            "data_root": str(data_root),
            "images_dir": str(images_dir),
            "ground_truth_dir": str(ground_truth_dir),
            "query_crops_dir": str(query_crops_dir),
        },
    }


def prepare_dataset(config: Dict[str, object], force_generate_demo: bool = False) -> Dict[str, object]:
    mode = str(config["dataset"].get("mode", "demo")).lower()
    if mode == "demo":
        return ensure_demo_dataset(config, force_generate=force_generate_demo)
    if mode == "oxford5k":
        return ensure_oxford5k_dataset(config)
    raise ValueError(f"Unsupported dataset mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Run visual bag-of-words image retrieval.")
    parser.add_argument("--config", default="Assignment2/config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Generate a deterministic demo dataset before running retrieval.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_bundle = prepare_dataset(config, force_generate_demo=args.generate_demo)
    outputs = bow_retrieval_pipeline(dataset_bundle, config)

    metrics = outputs["metrics"]
    dataset_mode = str(dataset_bundle.get("mode", "demo")).lower()
    print("Visual BoW retrieval completed.")
    print(f"Dataset mode: {dataset_mode}")
    print(f"Reference images: {int(metrics['reference_images'])}")
    print(f"Queries: {int(metrics['queries'])}")
    if dataset_mode == "oxford5k":
        if "map" in metrics:
            print(f"mAP: {metrics['map']:.3f}")
        for key in sorted(metrics):
            if key.startswith("top") and key.endswith("_success"):
                print(f"{key}: {metrics[key]:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")
    else:
        for key in sorted(metrics):
            if key.startswith("top") and key.endswith("_accuracy"):
                print(f"{key}: {metrics[key]:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")
    print(f"Results saved to: {config['output']['results_dir']}")


if __name__ == "__main__":
    main()
