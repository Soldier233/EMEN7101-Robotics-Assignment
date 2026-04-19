import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PointArray = np.ndarray


def load_image(image_path: str, color: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def create_feature_extractor(feature_type: str = "SIFT", max_features: int = 1000):
    feature_type = feature_type.upper()
    if feature_type == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is unavailable in this OpenCV build.")
        return cv2.SIFT_create(nfeatures=max_features), 128
    if feature_type == "ORB":
        return cv2.ORB_create(nfeatures=max_features), 32
    raise ValueError(f"Unsupported feature type: {feature_type}")


def extract_features(
    image: np.ndarray,
    feature_type: str = "SIFT",
    max_features: int = 1000,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    extractor, descriptor_dim = create_feature_extractor(feature_type, max_features)
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    if descriptors is None:
        dtype = np.uint8 if feature_type.upper() == "ORB" else np.float32
        descriptors = np.empty((0, descriptor_dim), dtype=dtype)
    return keypoints, descriptors


def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    feature_type: str = "SIFT",
    ratio_test: float = 0.75,
) -> List[cv2.DMatch]:
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    feature_type = feature_type.upper()
    if feature_type == "SIFT":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=64)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        desc1 = desc1.astype(np.float32, copy=False)
        desc2 = desc2.astype(np.float32, copy=False)
    elif feature_type == "ORB":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio_test * second.distance:
            good_matches.append(best)
    good_matches.sort(key=lambda match: match.distance)
    return good_matches


def matched_keypoints_to_points(
    keypoints1: Sequence[cv2.KeyPoint],
    keypoints2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
) -> Tuple[PointArray, PointArray]:
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float64)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float64)
    return points1, points2


def to_homogeneous(points: PointArray) -> PointArray:
    return np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])


def normalize_points(points: PointArray) -> Tuple[PointArray, np.ndarray]:
    if len(points) == 0:
        raise ValueError("At least one point is required for normalization.")

    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    distances = np.linalg.norm(shifted, axis=1)
    mean_distance = float(np.mean(distances))
    scale = 1.0 if mean_distance < 1e-12 else math.sqrt(2.0) / mean_distance

    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized_h = (transform @ to_homogeneous(points).T).T
    return normalized_h[:, :2], transform


def estimate_fundamental_eight_point(
    points1: PointArray,
    points2: PointArray,
    normalize: bool = True,
) -> np.ndarray:
    if len(points1) != len(points2):
        raise ValueError("Point arrays must have the same length.")
    if len(points1) < 8:
        raise ValueError("At least 8 correspondences are required.")

    if normalize:
        norm1, t1 = normalize_points(points1)
        norm2, t2 = normalize_points(points2)
    else:
        norm1, norm2 = points1.copy(), points2.copy()
        t1 = np.eye(3, dtype=np.float64)
        t2 = np.eye(3, dtype=np.float64)

    x1 = norm1[:, 0]
    y1 = norm1[:, 1]
    x2 = norm2[:, 0]
    y2 = norm2[:, 1]
    a = np.column_stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(len(norm1))])

    _, _, vt = np.linalg.svd(a)
    f = vt[-1].reshape(3, 3)

    u, s, vt = np.linalg.svd(f)
    s[-1] = 0.0
    f_rank2 = u @ np.diag(s) @ vt
    f_denorm = t2.T @ f_rank2 @ t1

    if abs(f_denorm[-1, -1]) > 1e-12:
        f_denorm /= f_denorm[-1, -1]
    else:
        norm = np.linalg.norm(f_denorm)
        if norm > 1e-12:
            f_denorm /= norm
    return f_denorm


def compute_sampson_distance(F: np.ndarray, points1: PointArray, points2: PointArray) -> np.ndarray:
    x1 = to_homogeneous(points1)
    x2 = to_homogeneous(points2)

    fx1 = (F @ x1.T).T
    ftx2 = (F.T @ x2.T).T
    numerator = np.sum(x2 * fx1, axis=1) ** 2
    denominator = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    denominator = np.maximum(denominator, 1e-12)
    return numerator / denominator


def estimate_fundamental_ransac(
    points1: PointArray,
    points2: PointArray,
    iterations: int = 2000,
    threshold: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if len(points1) != len(points2):
        raise ValueError("Point arrays must have the same length.")
    if len(points1) < 8:
        raise ValueError("At least 8 correspondences are required for RANSAC.")

    rng = np.random.default_rng(seed)
    best_f = None
    best_inliers = None
    best_score = -1
    best_error = float("inf")

    for _ in range(iterations):
        sample_indices = rng.choice(len(points1), size=8, replace=False)
        try:
            candidate_f = estimate_fundamental_eight_point(points1[sample_indices], points2[sample_indices], normalize=True)
        except np.linalg.LinAlgError:
            continue

        distances = compute_sampson_distance(candidate_f, points1, points2)
        inliers = distances < threshold
        inlier_count = int(np.sum(inliers))
        if inlier_count == 0:
            continue
        mean_error = float(np.mean(distances[inliers]))
        if inlier_count > best_score or (inlier_count == best_score and mean_error < best_error):
            best_score = inlier_count
            best_error = mean_error
            best_f = candidate_f
            best_inliers = inliers

    if best_f is None or best_inliers is None or np.sum(best_inliers) < 8:
        raise RuntimeError("RANSAC failed to estimate a valid fundamental matrix.")

    refined_f = estimate_fundamental_eight_point(points1[best_inliers], points2[best_inliers], normalize=True)
    refined_distances = compute_sampson_distance(refined_f, points1, points2)
    refined_inliers = refined_distances < threshold
    metrics = {
        "ransac_iterations": float(iterations),
        "ransac_threshold": float(threshold),
        "inlier_count": float(np.sum(refined_inliers)),
        "match_count": float(len(points1)),
        "inlier_ratio": float(np.sum(refined_inliers) / len(points1)),
        "mean_sampson_distance": float(np.mean(refined_distances[refined_inliers])) if np.any(refined_inliers) else float("inf"),
    }
    return refined_f, refined_inliers, metrics


def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    return K2.T @ F @ K1


def enforce_essential_constraints(E: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(E)
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1.0
    if np.linalg.det(vt) < 0:
        vt[-1, :] *= -1.0
    return u @ np.diag([1.0, 1.0, 0.0]) @ vt


def decompose_essential_matrix(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    u, _, vt = np.linalg.svd(E)
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1.0
    if np.linalg.det(vt) < 0:
        vt[-1, :] *= -1.0

    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t = u[:, 2]

    candidates = []
    for r in (u @ w @ vt, u @ w.T @ vt):
        if np.linalg.det(r) < 0:
            r *= -1.0
        candidates.append((r, t.copy()))
        candidates.append((r, -t.copy()))
    return candidates


def make_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t.reshape(3, 1)])


def triangulate_point_dlt(P1: np.ndarray, P2: np.ndarray, x1: Sequence[float], x2: Sequence[float]) -> np.ndarray:
    x1x, x1y = x1
    x2x, x2y = x2
    A = np.array(
        [
            x1x * P1[2] - P1[0],
            x1y * P1[2] - P1[1],
            x2x * P2[2] - P2[0],
            x2y * P2[2] - P2[1],
        ],
        dtype=np.float64,
    )
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    X /= X[3]
    return X[:3]


def triangulate_points(P1: np.ndarray, P2: np.ndarray, points1: PointArray, points2: PointArray) -> np.ndarray:
    points_3d = [triangulate_point_dlt(P1, P2, p1, p2) for p1, p2 in zip(points1, points2)]
    return np.asarray(points_3d, dtype=np.float64)


def count_points_in_front_of_cameras(R: np.ndarray, t: np.ndarray, points_3d: np.ndarray) -> int:
    depth1 = points_3d[:, 2]
    cam2_points = (R @ points_3d.T + t.reshape(3, 1)).T
    depth2 = cam2_points[:, 2]
    return int(np.sum((depth1 > 1e-6) & (depth2 > 1e-6)))


def select_correct_pose(
    candidates: Sequence[Tuple[np.ndarray, np.ndarray]],
    K1: np.ndarray,
    K2: np.ndarray,
    points1: PointArray,
    points2: PointArray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    P1 = make_projection_matrix(K1, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    best_result = None
    best_positive = -1
    for index, (R, t) in enumerate(candidates):
        P2 = make_projection_matrix(K2, R, t)
        points_3d = triangulate_points(P1, P2, points1, points2)
        positive_count = count_points_in_front_of_cameras(R, t, points_3d)
        if positive_count > best_positive:
            best_positive = positive_count
            best_result = (R, t, points_3d, index)

    if best_result is None:
        raise RuntimeError("Failed to select a valid camera pose.")

    R, t, points_3d, index = best_result
    metrics = {
        "selected_pose_index": float(index),
        "positive_depth_count": float(best_positive),
        "triangulated_count": float(len(points_3d)),
    }
    return R, t, points_3d, metrics


def _points_to_cv(points: PointArray) -> np.ndarray:
    return points.reshape(-1, 1, 2).astype(np.float32)


def draw_feature_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    keypoints1: Sequence[cv2.KeyPoint],
    keypoints2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    inlier_mask: Optional[np.ndarray],
    output_path: str,
    max_draw_matches: int = 80,
) -> str:
    if inlier_mask is not None:
        filtered_matches = [m for m, keep in zip(matches, inlier_mask) if bool(keep)]
    else:
        filtered_matches = list(matches)
    filtered_matches = filtered_matches[:max_draw_matches]

    image = cv2.drawMatches(
        image1,
        list(keypoints1),
        image2,
        list(keypoints2),
        filtered_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(output_path, image)
    return output_path


def draw_epipolar_lines(
    image1: np.ndarray,
    image2: np.ndarray,
    F: np.ndarray,
    points1: PointArray,
    points2: PointArray,
    output_path: str,
    max_lines: int = 20,
) -> str:
    if len(points1) == 0:
        raise ValueError("At least one correspondence is required to draw epipolar lines.")

    count = min(max_lines, len(points1))
    pts1 = points1[:count]
    pts2 = points2[:count]
    img1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR) if image1.ndim == 2 else image1.copy()
    img2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR) if image2.ndim == 2 else image2.copy()

    lines1 = cv2.computeCorrespondEpilines(_points_to_cv(pts2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(_points_to_cv(pts1), 1, F).reshape(-1, 3)

    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(count, 3))

    def draw_lines(img: np.ndarray, lines: np.ndarray, pts: PointArray) -> np.ndarray:
        h, w = img.shape[:2]
        for line, point, color in zip(lines, pts, colors):
            a, b, c = line
            color_tuple = tuple(int(v) for v in color.tolist())
            if abs(b) > 1e-12:
                x0, y0 = 0, int(round(-c / b))
                x1, y1 = w, int(round(-(c + a * w) / b))
            else:
                x = int(round(-c / max(a, 1e-12)))
                x0, y0 = x, 0
                x1, y1 = x, h
            cv2.line(img, (x0, y0), (x1, y1), color_tuple, 1)
            cv2.circle(img, tuple(np.round(point).astype(int)), 5, color_tuple, -1)
        return img

    vis1 = draw_lines(img1, lines1, pts1)
    vis2 = draw_lines(img2, lines2, pts2)
    combined = np.hstack([vis1, vis2])
    cv2.imwrite(output_path, combined)
    return output_path


def draw_reconstruction(points_3d: np.ndarray, output_path: str) -> Optional[str]:
    if len(points_3d) == 0:
        return None
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=10, c=points_3d[:, 2], cmap="viridis")
    ax.set_title("Triangulated 3D Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def rotation_error_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    delta = R_gt.T @ R_est
    trace_value = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_value)))


def translation_angle_error_deg(t_gt: np.ndarray, t_est: np.ndarray) -> float:
    gt = np.asarray(t_gt, dtype=np.float64).reshape(-1)
    est = np.asarray(t_est, dtype=np.float64).reshape(-1)
    gt /= max(np.linalg.norm(gt), 1e-12)
    est /= max(np.linalg.norm(est), 1e-12)
    cosine = np.clip(np.dot(gt, est), -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosine)))
    return min(angle, 180.0 - angle)


def _matrix_to_string(matrix: np.ndarray) -> str:
    return np.array2string(matrix, precision=6, suppress_small=True)


def save_pose_results(
    output_path: str,
    F: np.ndarray,
    E: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    metrics: Dict[str, float],
) -> str:
    lines = [
        "Camera Pose Estimation Results",
        "=" * 32,
        "",
        "Fundamental matrix F:",
        _matrix_to_string(F),
        "",
        "Essential matrix E:",
        _matrix_to_string(E),
        "",
        "Rotation matrix R:",
        _matrix_to_string(R),
        "",
        "Translation vector t:",
        _matrix_to_string(t.reshape(3, 1)),
        "",
        "Metrics:",
    ]
    for key in sorted(metrics):
        lines.append(f"- {key}: {metrics[key]}")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_error_analysis(
    output_path: str,
    metrics: Dict[str, float],
    pose_gt: Optional[Dict[str, np.ndarray]] = None,
    R_est: Optional[np.ndarray] = None,
    t_est: Optional[np.ndarray] = None,
) -> str:
    lines = ["Error Analysis", "=" * 14, ""]
    if pose_gt is not None and R_est is not None and t_est is not None:
        rot_err = rotation_error_deg(pose_gt["R"], R_est)
        trans_err = translation_angle_error_deg(pose_gt["t"], t_est)
        lines.extend(
            [
                f"Rotation error (deg): {rot_err:.6f}",
                f"Translation direction error (deg): {trans_err:.6f}",
                "",
            ]
        )
        metrics["rotation_error_deg"] = rot_err
        metrics["translation_angle_error_deg"] = trans_err
    else:
        lines.append("Ground truth pose not provided. Reporting reconstruction summary only.")
        lines.append("")

    for key in sorted(metrics):
        lines.append(f"- {key}: {metrics[key]}")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "K1" in data and "K2" in data:
        return data["K1"].astype(np.float64), data["K2"].astype(np.float64)
    if "K" in data:
        K = data["K"].astype(np.float64)
        return K, K.copy()
    raise ValueError(f"Intrinsics file must contain K or K1/K2 arrays: {path}")


def _load_ground_truth_pose(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path:
        return None
    data = np.load(path)
    return {
        "R": data["R"].astype(np.float64),
        "t": data["t"].astype(np.float64).reshape(3),
    }


def run_pose_estimation_pipeline(
    image1_path: str,
    image2_path: str,
    intrinsics_path: str,
    output_dir: str,
    feature_type: str = "SIFT",
    max_features: int = 1200,
    ratio_test: float = 0.75,
    ransac_iterations: int = 2000,
    ransac_threshold: float = 1.0,
    ground_truth_path: Optional[str] = None,
    random_seed: int = 42,
    save_reconstruction_plot: bool = True,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    image1_gray = load_image(image1_path, color=False)
    image2_gray = load_image(image2_path, color=False)
    image1_color = load_image(image1_path, color=True)
    image2_color = load_image(image2_path, color=True)

    keypoints1, descriptors1 = extract_features(image1_gray, feature_type=feature_type, max_features=max_features)
    keypoints2, descriptors2 = extract_features(image2_gray, feature_type=feature_type, max_features=max_features)
    matches = match_features(descriptors1, descriptors2, feature_type=feature_type, ratio_test=ratio_test)
    if len(matches) < 8:
        raise RuntimeError(f"Not enough feature matches after ratio test: {len(matches)}")

    points1, points2 = matched_keypoints_to_points(keypoints1, keypoints2, matches)
    F, inlier_mask, ransac_metrics = estimate_fundamental_ransac(
        points1,
        points2,
        iterations=ransac_iterations,
        threshold=ransac_threshold,
        seed=random_seed,
    )

    K1, K2 = _load_intrinsics(intrinsics_path)
    E = essential_from_fundamental(F, K1, K2)
    E = enforce_essential_constraints(E)
    candidates = decompose_essential_matrix(E)

    inlier_points1 = points1[inlier_mask]
    inlier_points2 = points2[inlier_mask]
    R, t, points_3d, pose_metrics = select_correct_pose(candidates, K1, K2, inlier_points1, inlier_points2)

    singular_values_f = np.linalg.svd(F, compute_uv=False)
    singular_values_e = np.linalg.svd(E, compute_uv=False)
    metrics: Dict[str, float] = {
        "feature_count_image1": float(len(keypoints1)),
        "feature_count_image2": float(len(keypoints2)),
        "raw_match_count": float(len(matches)),
        "fundamental_rank": float(np.sum(singular_values_f > 1e-8)),
        "essential_singular_value_1": float(singular_values_e[0]),
        "essential_singular_value_2": float(singular_values_e[1]),
        "essential_singular_value_3": float(singular_values_e[2]),
    }
    metrics.update(ransac_metrics)
    metrics.update(pose_metrics)

    pose_results_path = os.path.join(output_dir, "pose_results.txt")
    feature_matches_path = os.path.join(output_dir, "feature_matches.png")
    epipolar_lines_path = os.path.join(output_dir, "epipolar_lines.png")
    error_analysis_path = os.path.join(output_dir, "error_analysis.txt")
    reconstruction_path = os.path.join(output_dir, "reconstruction.png")
    metrics_json_path = os.path.join(output_dir, "metrics.json")

    draw_feature_matches(image1_color, image2_color, keypoints1, keypoints2, matches, inlier_mask, feature_matches_path)
    draw_epipolar_lines(image1_color, image2_color, F, inlier_points1, inlier_points2, epipolar_lines_path)
    if save_reconstruction_plot:
        draw_reconstruction(points_3d, reconstruction_path)

    save_pose_results(pose_results_path, F, E, R, t, metrics)
    ground_truth_pose = _load_ground_truth_pose(ground_truth_path)
    save_error_analysis(error_analysis_path, metrics, pose_gt=ground_truth_pose, R_est=R, t_est=t)
    Path(metrics_json_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "F": F,
        "E": E,
        "R": R,
        "t": t,
        "inliers": inlier_mask,
        "points_3d": points_3d,
        "metrics": metrics,
        "output_paths": {
            "pose_results": pose_results_path,
            "feature_matches": feature_matches_path,
            "epipolar_lines": epipolar_lines_path,
            "error_analysis": error_analysis_path,
            "reconstruction": reconstruction_path if save_reconstruction_plot else None,
            "metrics_json": metrics_json_path,
        },
    }
