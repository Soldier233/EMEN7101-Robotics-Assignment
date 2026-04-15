import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
    image_id: str
    label: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    histogram: Optional[np.ndarray] = None


@dataclass
class QueryEntry(ImageEntry):
    query_name: str = ""
    source_image_id: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None
    relevant_ids: Set[str] = field(default_factory=set)
    junk_ids: Set[str] = field(default_factory=set)


def load_image_paths(folder: str, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> List[str]:
    if not os.path.isdir(folder):
        return []
    paths = []
    lowered = tuple(ext.lower() for ext in extensions)
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(lowered):
            paths.append(os.path.join(folder, name))
    return paths


def infer_label_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0]


def infer_image_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


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


def collect_reference_entries(
    image_records: Sequence[Dict[str, object]],
    feature_type: str,
    max_features: int,
) -> List[ImageEntry]:
    entries = []
    for record in image_records:
        path = str(record["path"])
        keypoints, descriptors = extract_features(path, feature_type=feature_type, max_features=max_features)
        entries.append(
            ImageEntry(
                path=path,
                image_id=str(record.get("image_id", infer_image_id_from_path(path))),
                label=str(record.get("label", infer_label_from_path(path))),
                keypoints=keypoints,
                descriptors=descriptors,
            )
        )
    return entries


def collect_query_entries(
    query_records: Sequence[Dict[str, object]],
    feature_type: str,
    max_features: int,
) -> List[QueryEntry]:
    entries = []
    for record in query_records:
        path = str(record["query_path"])
        keypoints, descriptors = extract_features(path, feature_type=feature_type, max_features=max_features)
        bbox = record.get("bbox")
        bbox_tuple = tuple(int(v) for v in bbox) if bbox is not None else None
        entries.append(
            QueryEntry(
                path=path,
                image_id=str(record.get("query_id", infer_image_id_from_path(path))),
                label=str(record.get("label", infer_label_from_path(path))),
                keypoints=keypoints,
                descriptors=descriptors,
                query_name=str(record.get("query_name", infer_image_id_from_path(path))),
                source_image_id=str(record.get("source_image_id", "")),
                bbox=bbox_tuple,
                relevant_ids=set(str(item) for item in record.get("relevant_ids", [])),
                junk_ids=set(str(item) for item in record.get("junk_ids", [])),
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


def compute_geometric_verification_score(
    query_entry: QueryEntry,
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
    query_entry: QueryEntry,
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


def compute_average_precision(full_results: Sequence[Dict[str, object]], relevant_ids: Set[str], junk_ids: Set[str]) -> float:
    if not relevant_ids:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    filtered_rank = 0
    for result in full_results:
        image_id = str(result["image_id"])
        if image_id in junk_ids:
            continue
        filtered_rank += 1
        if image_id in relevant_ids:
            hit_count += 1
            precision_sum += hit_count / filtered_rank

    return precision_sum / len(relevant_ids)


def compute_first_relevant_rank(full_results: Sequence[Dict[str, object]], relevant_ids: Set[str], junk_ids: Set[str]) -> Optional[int]:
    filtered_rank = 0
    for result in full_results:
        image_id = str(result["image_id"])
        if image_id in junk_ids:
            continue
        filtered_rank += 1
        if image_id in relevant_ids:
            return filtered_rank
    return None


def evaluate_demo_rankings(
    rankings: Sequence[Dict[str, object]],
    metric_top_k: Sequence[int],
) -> Dict[str, float]:
    total = max(len(rankings), 1)
    reciprocal_ranks = []
    hit_counts = {k: 0 for k in metric_top_k}

    for item in rankings:
        query_label = item["query_label"]
        found_rank = None
        for result in item["full_results"]:
            if result["label"] == query_label:
                found_rank = int(result["rank"])
                break

        reciprocal_ranks.append(0.0 if found_rank is None else 1.0 / found_rank)
        for k in metric_top_k:
            if found_rank is not None and found_rank <= k:
                hit_counts[k] += 1

    metrics: Dict[str, float] = {
        "queries": float(len(rankings)),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }
    for k in metric_top_k:
        metrics[f"top{k}_accuracy"] = hit_counts[k] / total
    return metrics


def evaluate_oxford_rankings(
    rankings: Sequence[Dict[str, object]],
    metric_top_k: Sequence[int],
) -> Dict[str, float]:
    total = max(len(rankings), 1)
    ap_values = []
    reciprocal_ranks = []
    hit_counts = {k: 0 for k in metric_top_k}

    for item in rankings:
        relevant_ids = set(item.get("relevant_ids", []))
        junk_ids = set(item.get("junk_ids", []))
        full_results = item["full_results"]
        ap = compute_average_precision(full_results, relevant_ids, junk_ids)
        ap_values.append(ap)

        first_relevant_rank = compute_first_relevant_rank(full_results, relevant_ids, junk_ids)
        reciprocal_ranks.append(0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank)
        for k in metric_top_k:
            if first_relevant_rank is not None and first_relevant_rank <= k:
                hit_counts[k] += 1

    metrics = {
        "queries": float(len(rankings)),
        "map": float(np.mean(ap_values)) if ap_values else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }
    for k in metric_top_k:
        metrics[f"top{k}_success"] = hit_counts[k] / total
    return metrics


def evaluate_rankings(
    rankings: Sequence[Dict[str, object]],
    dataset_mode: str,
    evaluation_cfg: Dict[str, object],
    retrieval_top_k: int,
) -> Dict[str, float]:
    metric_top_k = sorted({int(value) for value in evaluation_cfg.get("top_k_values", [retrieval_top_k]) if int(value) > 0})
    if not metric_top_k:
        metric_top_k = [retrieval_top_k]

    if dataset_mode == "oxford5k":
        return evaluate_oxford_rankings(rankings, metric_top_k=metric_top_k)
    return evaluate_demo_rankings(rankings, metric_top_k=metric_top_k)


def save_rankings_csv(rankings: Sequence[Dict[str, object]], output_csv: str, dataset_mode: str) -> None:
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_name",
                "query_path",
                "query_label",
                "query_image_id",
                "rank",
                "match_path",
                "match_label",
                "match_image_id",
                "score",
                "is_relevant",
                "is_junk",
            ]
        )
        for item in rankings:
            relevant_ids = set(item.get("relevant_ids", []))
            junk_ids = set(item.get("junk_ids", []))
            for result in item["full_results"]:
                image_id = str(result["image_id"])
                writer.writerow(
                    [
                        item.get("query_name", ""),
                        item["query_path"],
                        item["query_label"],
                        item.get("query_image_id", ""),
                        result["rank"],
                        result["path"],
                        result["label"],
                        image_id,
                        f"{result['score']:.6f}",
                        int(image_id in relevant_ids) if dataset_mode == "oxford5k" else "",
                        int(image_id in junk_ids) if dataset_mode == "oxford5k" else "",
                    ]
                )


def save_metrics_txt(metrics: Dict[str, object], output_txt: str) -> None:
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
    dataset_mode: str,
    max_queries: int = 5,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for item in rankings[:max_queries]:
        cols = 1 + len(item["results"])
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
        if cols == 1:
            axes = [axes]

        query_title = item.get("query_name", item["query_label"])
        _plot_match(axes[0], item["query_path"], f"Query\n{query_title}")
        relevant_ids = set(item.get("relevant_ids", []))
        junk_ids = set(item.get("junk_ids", []))
        for idx, result in enumerate(item["results"], start=1):
            suffix = ""
            if dataset_mode == "oxford5k":
                if result["image_id"] in relevant_ids:
                    suffix = "\nrelevant"
                elif result["image_id"] in junk_ids:
                    suffix = "\njunk"
            _plot_match(
                axes[idx],
                result["path"],
                f"Rank {result['rank']}\n{result['image_id']} ({result['score']:.2f}){suffix}",
            )

        fig.tight_layout()
        stem = os.path.splitext(os.path.basename(item["query_path"]))[0]
        fig.savefig(os.path.join(output_dir, f"{stem}_ranking.png"), dpi=180)
        plt.close(fig)


def bow_retrieval_pipeline(dataset_bundle: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    dataset_mode = str(dataset_bundle.get("mode", "demo")).lower()
    feature_cfg = config["feature"]
    vocab_cfg = config["vocabulary"]
    retrieval_cfg = config["retrieval"]
    evaluation_cfg = config.get("evaluation", {})
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

    reference_entries = collect_reference_entries(dataset_bundle["reference_images"], feature_type, max_features)
    query_entries = collect_query_entries(dataset_bundle["queries"], feature_type, max_features)

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

        full_results = []
        for rank, idx in enumerate(ranking, start=1):
            idx = int(idx)
            reference_entry = reference_entries[idx]
            full_results.append(
                {
                    "rank": rank,
                    "path": reference_entry.path,
                    "label": reference_entry.label,
                    "image_id": reference_entry.image_id,
                    "score": float(scores[idx]),
                }
            )

        rankings.append(
            {
                "query_name": entry.query_name,
                "query_path": entry.path,
                "query_label": entry.label,
                "query_image_id": entry.image_id,
                "source_image_id": entry.source_image_id,
                "bbox": entry.bbox,
                "relevant_ids": sorted(entry.relevant_ids),
                "junk_ids": sorted(entry.junk_ids),
                "results": full_results[:top_k],
                "full_results": full_results,
            }
        )

    metrics = evaluate_rankings(
        rankings,
        dataset_mode=dataset_mode,
        evaluation_cfg=evaluation_cfg,
        retrieval_top_k=top_k,
    )
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
    save_rankings_csv(rankings, os.path.join(output_cfg["results_dir"], "retrieval_results.csv"), dataset_mode)
    save_metrics_txt(metrics, os.path.join(output_cfg["results_dir"], "metrics.txt"))

    if output_cfg.get("save_visualizations", True):
        save_visualizations(rankings, os.path.join(output_cfg["results_dir"], "visualizations"), dataset_mode)

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
