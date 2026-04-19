import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pose_estimation import run_pose_estimation_pipeline


ASSIGNMENT_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_DATA_DIR = ASSIGNMENT_DIR / "test_data"
DEFAULT_RESULTS_DIR = ASSIGNMENT_DIR / "results"


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    camera_points = (R @ points_3d.T + t.reshape(3, 1)).T
    valid = camera_points[:, 2] > 1e-6
    projected = (K @ camera_points.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return projected, valid


def draw_projected_scene(
    image_size: Tuple[int, int],
    points_2d: np.ndarray,
    valid_mask: np.ndarray,
    depths: np.ndarray,
    seed: int,
) -> np.ndarray:
    height, width = image_size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    grid_step = 40
    for x in range(0, width, grid_step):
        cv2.line(image, (x, 0), (x, height - 1), (230, 230, 230), 1)
    for y in range(0, height, grid_step):
        cv2.line(image, (0, y), (width - 1, y), (230, 230, 230), 1)

    for idx, (point, is_valid, depth) in enumerate(zip(points_2d, valid_mask, depths)):
        if not is_valid:
            continue
        x, y = np.round(point).astype(int)
        if x < 12 or x >= width - 12 or y < 12 or y >= height - 12:
            continue
        radius = int(4 + (idx % 3))
        shade = int(30 + min(max((depth - 3.0) * 15.0, 0.0), 140.0))
        color = (shade, 20 + (idx * 17) % 180, 255 - shade // 2)
        cv2.circle(image, (x, y), radius, color, -1)
        cv2.circle(image, (x, y), radius + 3, (0, 0, 0), 1)
        if idx % 5 == 0:
            offset = rng.integers(-10, 10, size=2)
            endpoint = (int(x + offset[0]), int(y + offset[1]))
            cv2.line(image, (x, y), endpoint, (0, 0, 0), 1)
        if idx % 7 == 0:
            cv2.rectangle(image, (x - 8, y - 8), (x + 8, y + 8), (0, 0, 0), 1)

    return image


def generate_synthetic_correspondence_dataset(output_dir: Path, overwrite: bool = False) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image1_path = output_dir / "image1.png"
    image2_path = output_dir / "image2.png"
    intrinsics_path = output_dir / "intrinsics.npz"
    ground_truth_path = output_dir / "ground_truth_pose.npz"
    metadata_path = output_dir / "scene_metadata.npz"

    if (
        not overwrite
        and image1_path.exists()
        and image2_path.exists()
        and intrinsics_path.exists()
        and ground_truth_path.exists()
        and metadata_path.exists()
    ):
        return {
            "image1": str(image1_path),
            "image2": str(image2_path),
            "intrinsics": str(intrinsics_path),
            "ground_truth": str(ground_truth_path),
            "metadata": str(metadata_path),
        }

    width, height = 960, 720
    fx = fy = 820.0
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    rng = np.random.default_rng(42)
    plane_xy = rng.uniform([-1.4, -1.0], [1.4, 1.0], size=(36, 2))
    plane_z = rng.uniform(4.0, 7.5, size=(36, 1))
    cluster_xy = rng.uniform([-0.9, -0.7], [0.9, 0.7], size=(24, 2))
    cluster_z = rng.uniform(5.5, 9.0, size=(24, 1))
    points_3d = np.vstack([
        np.hstack([plane_xy, plane_z]),
        np.hstack([cluster_xy, cluster_z]),
    ]).astype(np.float64)

    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros(3, dtype=np.float64)
    R2 = rotation_matrix_y(np.deg2rad(9.0))
    t2 = np.array([0.45, 0.03, 0.02], dtype=np.float64)

    points1, valid1 = project_points(K, R1, t1, points_3d)
    points2, valid2 = project_points(K, R2, t2, points_3d)
    valid = valid1 & valid2
    valid &= (points1[:, 0] >= 12) & (points1[:, 0] < width - 12) & (points1[:, 1] >= 12) & (points1[:, 1] < height - 12)
    valid &= (points2[:, 0] >= 12) & (points2[:, 0] < width - 12) & (points2[:, 1] >= 12) & (points2[:, 1] < height - 12)

    points1 += rng.normal(0.0, 0.6, size=points1.shape)
    points2 += rng.normal(0.0, 0.6, size=points2.shape)

    image1 = draw_projected_scene((height, width), points1, valid, points_3d[:, 2], seed=1)
    image2 = draw_projected_scene((height, width), points2, valid, (R2 @ points_3d.T + t2.reshape(3, 1)).T[:, 2], seed=2)

    cv2.imwrite(str(image1_path), image1)
    cv2.imwrite(str(image2_path), image2)
    np.savez(intrinsics_path, K=K)
    np.savez(ground_truth_path, R=R2, t=t2)
    np.savez(
        metadata_path,
        points_3d=points_3d,
        image1_points=points1,
        image2_points=points2,
        valid_mask=valid,
    )

    return {
        "image1": str(image1_path),
        "image2": str(image2_path),
        "intrinsics": str(intrinsics_path),
        "ground_truth": str(ground_truth_path),
        "metadata": str(metadata_path),
    }


def _load_intrinsics_array(path: str) -> np.ndarray:
    data = np.load(path)
    if "K" in data:
        return data["K"].astype(np.float64)
    if "K1" in data:
        return data["K1"].astype(np.float64)
    raise ValueError(f"Intrinsics file must contain K or K1: {path}")


def resolve_inputs(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    if args.image1 and args.image2:
        if args.intrinsics:
            intrinsics = args.intrinsics
        elif args.k1 and args.k2:
            k1 = _load_intrinsics_array(args.k1)
            k2 = _load_intrinsics_array(args.k2)
            combined_path = str(Path(args.output_dir) / "_combined_intrinsics_tmp.npz")
            np.savez(combined_path, K1=k1, K2=k2)
            intrinsics = combined_path
        else:
            raise ValueError("Provide --intrinsics or both --k1 and --k2 when using custom images.")
        return {
            "image1": args.image1,
            "image2": args.image2,
            "intrinsics": intrinsics,
            "ground_truth": args.ground_truth,
        }

    dataset = generate_synthetic_correspondence_dataset(Path(args.test_data_dir), overwrite=args.regenerate_test_data)
    return {
        "image1": dataset["image1"],
        "image2": dataset["image2"],
        "intrinsics": dataset["intrinsics"],
        "ground_truth": args.ground_truth or dataset["ground_truth"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run camera pose estimation via epipolar geometry.")
    parser.add_argument("--image1", type=str, help="Path to the first image.")
    parser.add_argument("--image2", type=str, help="Path to the second image.")
    parser.add_argument("--intrinsics", type=str, help="Path to a .npz file containing K or K1/K2.")
    parser.add_argument("--k1", type=str, help="Path to first camera intrinsics .npz file.")
    parser.add_argument("--k2", type=str, help="Path to second camera intrinsics .npz file.")
    parser.add_argument("--ground-truth", type=str, default=None, help="Optional ground-truth pose .npz file.")
    parser.add_argument("--feature-type", type=str, default="SIFT", choices=["SIFT", "ORB"], help="Feature detector type.")
    parser.add_argument("--max-features", type=int, default=1200, help="Maximum features to extract.")
    parser.add_argument("--ratio-test", type=float, default=0.75, help="Lowe ratio test threshold.")
    parser.add_argument("--ransac-iterations", type=int, default=2000, help="Number of RANSAC iterations.")
    parser.add_argument("--ransac-threshold", type=float, default=1.0, help="Sampson distance inlier threshold.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directory for output artifacts.")
    parser.add_argument("--test-data-dir", type=str, default=str(DEFAULT_TEST_DATA_DIR), help="Directory for synthetic test data.")
    parser.add_argument("--regenerate-test-data", action="store_true", help="Regenerate the synthetic dataset before running.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for RANSAC.")
    parser.add_argument("--no-reconstruction-plot", action="store_true", help="Skip reconstruction plot generation.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    inputs = resolve_inputs(args)
    results = run_pose_estimation_pipeline(
        image1_path=inputs["image1"],
        image2_path=inputs["image2"],
        intrinsics_path=inputs["intrinsics"],
        output_dir=args.output_dir,
        feature_type=args.feature_type,
        max_features=args.max_features,
        ratio_test=args.ratio_test,
        ransac_iterations=args.ransac_iterations,
        ransac_threshold=args.ransac_threshold,
        ground_truth_path=inputs["ground_truth"],
        random_seed=args.seed,
        save_reconstruction_plot=not args.no_reconstruction_plot,
    )

    metrics = results["metrics"]
    print("Pose estimation completed.")
    print(f"Image 1: {inputs['image1']}")
    print(f"Image 2: {inputs['image2']}")
    print(f"Inliers: {int(metrics['inlier_count'])}/{int(metrics['raw_match_count'])}")
    print(f"Selected pose index: {int(metrics['selected_pose_index'])}")
    print(f"Positive depth points: {int(metrics['positive_depth_count'])}")
    if "rotation_error_deg" in metrics:
        print(f"Rotation error (deg): {metrics['rotation_error_deg']:.6f}")
    if "translation_angle_error_deg" in metrics:
        print(f"Translation direction error (deg): {metrics['translation_angle_error_deg']:.6f}")
    print("Output files:")
    for key, value in results["output_paths"].items():
        if value:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
