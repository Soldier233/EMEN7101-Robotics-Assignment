import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class ImageEntry:
    path: str
    label: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    histogram: Optional[np.ndarray] = None


def load_image_paths(folder: str, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> List[str]:
    if not os.path.isdir(folder):
        return []
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(tuple(ext.lower() for ext in extensions)):
            paths.append(os.path.join(folder, name))
    return paths


def infer_label_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0]


def _create_feature_extractor(feature_type: str, max_features: int):
    feature_type = feature_type.upper()
    if feature_type == "SIFT":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=max_features), 128
        raise RuntimeError("SIFT is unavailable in this OpenCV build.")
    if feature_type == "ORB":
        return cv2.ORB_create(nfeatures=max_features), 32
    raise ValueError(f"Unsupported feature type: {feature_type}")


def extract_features(
    image_path: str,
    feature_type: str = "SIFT",
    max_features: int = 800,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    extractor, descriptor_dim = _create_feature_extractor(feature_type, max_features)
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = np.empty((0, descriptor_dim), dtype=np.float32)
    descriptors = descriptors.astype(np.float32, copy=False)
    return keypoints, descriptors


def collect_image_entries(
    image_paths: Sequence[str],
    feature_type: str,
    max_features: int,
) -> List[ImageEntry]:
    entries = []
    for path in image_paths:
        keypoints, descriptors = extract_features(path, feature_type=feature_type, max_features=max_features)
        entries.append(
            ImageEntry(
                path=path,
                label=infer_label_from_path(path),
                keypoints=keypoints,
                descriptors=descriptors,
            )
        )
    return entries


def build_vocabulary(
    descriptor_sets: Sequence[np.ndarray],
    vocab_size: int = 64,
    max_descriptors: int = 30000,
    random_state: int = 42,
) -> np.ndarray:
    valid = [d for d in descriptor_sets if d is not None and len(d) > 0]
    if not valid:
        raise ValueError("No descriptors were extracted. Vocabulary cannot be built.")

    all_descriptors = np.vstack(valid).astype(np.float32, copy=False)
    if len(all_descriptors) < vocab_size:
        raise ValueError(
            f"Vocabulary size {vocab_size} exceeds the number of descriptors {len(all_descriptors)}."
        )

    if len(all_descriptors) > max_descriptors:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(all_descriptors), size=max_descriptors, replace=False)
        all_descriptors = all_descriptors[indices]

    model = MiniBatchKMeans(
        n_clusters=vocab_size,
        random_state=random_state,
        batch_size=min(4096, max(vocab_size * 16, 256)),
        n_init="auto",
    )
    model.fit(all_descriptors)
    return model.cluster_centers_.astype(np.float32)


def assign_visual_words(descriptors: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    if descriptors is None or len(descriptors) == 0:
        return np.empty((0,), dtype=np.int32)

    diff = descriptors[:, None, :] - vocabulary[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def compute_idf(assignments_per_image: Sequence[np.ndarray], vocab_size: int) -> np.ndarray:
    image_count = max(len(assignments_per_image), 1)
    doc_freq = np.zeros(vocab_size, dtype=np.float32)
    for words in assignments_per_image:
        if len(words) == 0:
            continue
        doc_freq[np.unique(words)] += 1.0
    return np.log((image_count + 1.0) / (doc_freq + 1.0)) + 1.0


def image_to_bow_histogram(
    descriptors: np.ndarray,
    vocabulary: np.ndarray,
    idf: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    vocab_size = vocabulary.shape[0]
    words = assign_visual_words(descriptors, vocabulary)
    histogram = np.bincount(words, minlength=vocab_size).astype(np.float32)

    if histogram.sum() > 0:
        histogram /= histogram.sum()

    if idf is not None:
        histogram *= idf.astype(np.float32, copy=False)

    if normalize:
        norm = np.linalg.norm(histogram)
        if norm > 1e-12:
            histogram /= norm
    return histogram


def compute_similarity(
    query_hist: np.ndarray,
    database_hists: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    metric = metric.lower()
    eps = 1e-10

    if metric == "cosine":
        db_norm = np.linalg.norm(database_hists, axis=1) + eps
        q_norm = np.linalg.norm(query_hist) + eps
        return (database_hists @ query_hist) / (db_norm * q_norm)

    if metric == "l1":
        return -np.sum(np.abs(database_hists - query_hist[None, :]), axis=1)

    if metric == "l2":
        return -np.linalg.norm(database_hists - query_hist[None, :], axis=1)

    if metric in {"chi2", "chi-squared", "chisquared"}:
        numerator = (database_hists - query_hist[None, :]) ** 2
        denominator = database_hists + query_hist[None, :] + eps
        return -0.5 * np.sum(numerator / denominator, axis=1)

    raise ValueError(f"Unsupported similarity metric: {metric}")


def search_image(
    query_hist: np.ndarray,
    database_hists: np.ndarray,
    image_paths: Sequence[str],
    labels: Sequence[str],
    top_k: int = 5,
    metric: str = "cosine",
) -> List[Dict[str, object]]:
    scores = compute_similarity(query_hist, database_hists, metric=metric)
    ranking = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(ranking, start=1):
        results.append(
            {
                "rank": rank,
                "path": image_paths[idx],
                "label": labels[idx],
                "score": float(scores[idx]),
            }
        )
    return results


def compute_geometric_verification_score(
    query_entry: ImageEntry,
    reference_entry: ImageEntry,
    ratio_test: float = 0.75,
    ransac_threshold: float = 4.0,
) -> float:
    if len(query_entry.descriptors) < 2 or len(reference_entry.descriptors) < 2:
        return 0.0

    norm_type = cv2.NORM_L2 if query_entry.descriptors.shape[1] > 32 else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = matcher.knnMatch(query_entry.descriptors, reference_entry.descriptors, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio_test * second.distance:
            good_matches.append(best)

    if len(good_matches) < 4:
        return float(len(good_matches))

    src_pts = np.float32([query_entry.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([reference_entry.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    inliers = int(mask.sum()) if mask is not None else 0
    return float(inliers) + 0.01 * len(good_matches)


def rerank_with_spatial_verification(
    query_entry: ImageEntry,
    reference_entries: Sequence[ImageEntry],
    scores: np.ndarray,
    ranking: np.ndarray,
    rerank_top_n: int = 5,
    ratio_test: float = 0.75,
    ransac_threshold: float = 4.0,
) -> np.ndarray:
    candidate_count = min(rerank_top_n, len(ranking))
    candidates = ranking[:candidate_count]
    verified = []
    for idx in candidates:
        verify_score = compute_geometric_verification_score(
            query_entry,
            reference_entries[int(idx)],
            ratio_test=ratio_test,
            ransac_threshold=ransac_threshold,
        )
        verified.append((verify_score, float(scores[int(idx)]), int(idx)))

    if verified and max(item[0] for item in verified) > 0:
        verified.sort(key=lambda item: (item[0], item[1]), reverse=True)
        reordered = [item[2] for item in verified]
        return np.asarray(reordered + ranking[candidate_count:].tolist(), dtype=np.int32)
    return ranking


def detect_loop_closure(
    histograms: Sequence[np.ndarray],
    threshold: float = 0.75,
    min_gap: int = 3,
    metric: str = "cosine",
) -> List[Dict[str, object]]:
    detections = []
    if len(histograms) <= min_gap:
        return detections

    for current_idx in range(min_gap, len(histograms)):
        history = np.asarray(histograms[: current_idx - min_gap + 1], dtype=np.float32)
        scores = compute_similarity(histograms[current_idx], history, metric=metric)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= threshold:
            detections.append(
                {
                    "current_index": current_idx,
                    "matched_index": best_idx,
                    "score": best_score,
                }
            )
    return detections


def evaluate_rankings(rankings: Sequence[Dict[str, object]], top_k: int) -> Dict[str, float]:
    total = max(len(rankings), 1)
    top1_hits = 0
    topk_hits = 0
    reciprocal_ranks = []

    for item in rankings:
        query_label = item["query_label"]
        ranked = item["results"]
        found_rank = None
        for result in ranked:
            if result["label"] == query_label:
                found_rank = int(result["rank"])
                break
        if found_rank == 1:
            top1_hits += 1
        if found_rank is not None and found_rank <= top_k:
            topk_hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    return {
        "queries": float(len(rankings)),
        "top1_accuracy": top1_hits / total,
        f"top{top_k}_accuracy": topk_hits / total,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def save_rankings_csv(rankings: Sequence[Dict[str, object]], output_csv: str) -> None:
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["query_path", "query_label", "rank", "match_path", "match_label", "score"]
        )
        for item in rankings:
            for result in item["results"]:
                writer.writerow(
                    [
                        item["query_path"],
                        item["query_label"],
                        result["rank"],
                        result["path"],
                        result["label"],
                        f"{result['score']:.6f}",
                    ]
                )


def save_metrics_txt(metrics: Dict[str, float], output_txt: str) -> None:
    with open(output_txt, "w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            if isinstance(value, float):
                handle.write(f"{key}: {value:.4f}\n")
            else:
                handle.write(f"{key}: {value}\n")


def _plot_match(ax, image_path: str, title: str) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        ax.text(0.5, 0.5, "Missing image", ha="center", va="center")
        ax.axis("off")
        return
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_visualizations(
    rankings: Sequence[Dict[str, object]],
    output_dir: str,
    max_queries: int = 5,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for item in rankings[:max_queries]:
        cols = 1 + len(item["results"])
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
        if cols == 1:
            axes = [axes]

        _plot_match(axes[0], item["query_path"], f"Query\n{item['query_label']}")
        for idx, result in enumerate(item["results"], start=1):
            _plot_match(
                axes[idx],
                result["path"],
                f"Rank {result['rank']}\n{result['label']} ({result['score']:.2f})",
            )

        fig.tight_layout()
        stem = os.path.splitext(os.path.basename(item["query_path"]))[0]
        fig.savefig(os.path.join(output_dir, f"{stem}_ranking.png"), dpi=180)
        plt.close(fig)


def bow_retrieval_pipeline(
    reference_paths: Sequence[str],
    query_paths: Sequence[str],
    config: Dict[str, object],
) -> Dict[str, object]:
    feature_cfg = config["feature"]
    vocab_cfg = config["vocabulary"]
    retrieval_cfg = config["retrieval"]
    loop_cfg = config.get("loop_closure", {})
    output_cfg = config["output"]
    spatial_cfg = config.get("spatial_verification", {})

    feature_type = feature_cfg.get("type", "SIFT")
    max_features = int(feature_cfg.get("max_features", 800))
    vocab_size = int(vocab_cfg.get("size", 64))
    max_descriptors = int(vocab_cfg.get("max_descriptors", 30000))
    random_state = int(vocab_cfg.get("random_state", 42))
    top_k = int(retrieval_cfg.get("top_k", 5))
    metric = retrieval_cfg.get("metric", "cosine")

    reference_entries = collect_image_entries(reference_paths, feature_type, max_features)
    query_entries = collect_image_entries(query_paths, feature_type, max_features)

    vocabulary = build_vocabulary(
        [entry.descriptors for entry in reference_entries],
        vocab_size=vocab_size,
        max_descriptors=max_descriptors,
        random_state=random_state,
    )

    reference_assignments = [assign_visual_words(entry.descriptors, vocabulary) for entry in reference_entries]
    idf = compute_idf(reference_assignments, vocab_size=vocabulary.shape[0])

    for entry in reference_entries:
        entry.histogram = image_to_bow_histogram(entry.descriptors, vocabulary, idf=idf)
    for entry in query_entries:
        entry.histogram = image_to_bow_histogram(entry.descriptors, vocabulary, idf=idf)

    database_hists = np.asarray([entry.histogram for entry in reference_entries], dtype=np.float32)
    database_paths = [entry.path for entry in reference_entries]
    database_labels = [entry.label for entry in reference_entries]

    rankings = []
    for entry in query_entries:
        scores = compute_similarity(entry.histogram, database_hists, metric=metric)
        ranking = np.argsort(scores)[::-1]
        if spatial_cfg.get("enabled", False):
            ranking = rerank_with_spatial_verification(
                entry,
                reference_entries,
                scores,
                ranking,
                rerank_top_n=int(spatial_cfg.get("rerank_top_n", 5)),
                ratio_test=float(spatial_cfg.get("ratio_test", 0.75)),
                ransac_threshold=float(spatial_cfg.get("ransac_threshold", 4.0)),
            )

        results = []
        for rank, idx in enumerate(ranking[:top_k], start=1):
            idx = int(idx)
            results.append(
                {
                    "rank": rank,
                    "path": database_paths[idx],
                    "label": database_labels[idx],
                    "score": float(scores[idx]),
                }
            )

        rankings.append(
            {
                "query_path": entry.path,
                "query_label": entry.label,
                "results": results,
            }
        )

    metrics = evaluate_rankings(rankings, top_k=top_k)
    metrics["reference_images"] = float(len(reference_entries))
    metrics["average_reference_keypoints"] = float(
        np.mean([len(entry.keypoints) for entry in reference_entries]) if reference_entries else 0.0
    )
    metrics["average_query_keypoints"] = float(
        np.mean([len(entry.keypoints) for entry in query_entries]) if query_entries else 0.0
    )

    os.makedirs(output_cfg["results_dir"], exist_ok=True)
    np.save(os.path.join(output_cfg["results_dir"], "vocabulary.npy"), vocabulary)
    np.save(os.path.join(output_cfg["results_dir"], "idf.npy"), idf)
    np.save(os.path.join(output_cfg["results_dir"], "database_histograms.npy"), database_hists)
    save_rankings_csv(rankings, os.path.join(output_cfg["results_dir"], "retrieval_results.csv"))
    save_metrics_txt(metrics, os.path.join(output_cfg["results_dir"], "metrics.txt"))

    if output_cfg.get("save_visualizations", True):
        save_visualizations(rankings, os.path.join(output_cfg["results_dir"], "visualizations"))

    loop_detections = []
    if loop_cfg.get("enabled", False):
        loop_detections = detect_loop_closure(
            [entry.histogram for entry in query_entries],
            threshold=float(loop_cfg.get("threshold", 0.75)),
            min_gap=int(loop_cfg.get("min_gap", 2)),
            metric=metric,
        )
        loop_path = os.path.join(output_cfg["results_dir"], "loop_closure.csv")
        with open(loop_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["current_index", "matched_index", "score"])
            writer.writeheader()
            writer.writerows(loop_detections)

    return {
        "vocabulary": vocabulary,
        "idf": idf,
        "rankings": rankings,
        "metrics": metrics,
        "loop_detections": loop_detections,
    }
