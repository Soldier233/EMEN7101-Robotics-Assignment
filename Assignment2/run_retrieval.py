import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

from bow_retrieval import IMAGE_EXTENSIONS, bow_retrieval_pipeline, load_image_paths


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


def ensure_dataset(config: Dict[str, object], force_generate: bool = False) -> Tuple[List[str], List[str]]:
    dataset_cfg = config["dataset"]
    reference_dir = dataset_cfg["reference_dir"]
    query_dir = dataset_cfg["query_dir"]

    reference_paths = load_image_paths(reference_dir, IMAGE_EXTENSIONS)
    query_paths = load_image_paths(query_dir, IMAGE_EXTENSIONS)

    if force_generate or not reference_paths or not query_paths:
        generate_demo_dataset(reference_dir, query_dir)
        reference_paths = load_image_paths(reference_dir, IMAGE_EXTENSIONS)
        query_paths = load_image_paths(query_dir, IMAGE_EXTENSIONS)

    if not reference_paths or not query_paths:
        raise RuntimeError("Dataset folders are empty even after demo generation.")
    return reference_paths, query_paths


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

    reference_paths, query_paths = ensure_dataset(config, force_generate=args.generate_demo)
    outputs = bow_retrieval_pipeline(reference_paths, query_paths, config)

    metrics = outputs["metrics"]
    print("Visual BoW retrieval completed.")
    print(f"Reference images: {int(metrics['reference_images'])}")
    print(f"Queries: {int(metrics['queries'])}")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.3f}")
    top_k = int(config["retrieval"]["top_k"])
    print(f"Top-{top_k} accuracy: {metrics[f'top{top_k}_accuracy']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
    print(f"Results saved to: {config['output']['results_dir']}")


if __name__ == "__main__":
    main()
