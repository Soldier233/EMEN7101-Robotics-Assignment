import argparse
import csv
import os
import time
import copy
import numpy as np
import open3d as o3d

from icp_registration import run_icp_mode


def random_rotation(max_deg, rng):
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R


def transform_points(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_error_deg(R_est, R_gt):
    R = R_est @ R_gt.T
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def load_dataset_points(path, target_count=12000):
    ext = os.path.splitext(path)[1].lower()
    pcd = o3d.geometry.PointCloud()

    if ext in [".ply", ".obj", ".stl", ".off"]:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.vertices) == 0:
            raise ValueError(f"Cannot read mesh vertices from {path}")
        mesh.compute_vertex_normals()
        pcd.points = mesh.vertices
    elif ext in [".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"]:
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise ValueError(f"Cannot read point cloud points from {path}")
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    points = np.asarray(pcd.points)
    min_b = points.min(axis=0)
    max_b = points.max(axis=0)
    extent = float(np.linalg.norm(max_b - min_b))
    if extent <= 0:
        raise ValueError(f"Degenerate dataset extent for {path}")

    voxel = max(extent / np.cbrt(max(target_count, 1000)), 1e-6)
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel)
    if len(pcd_ds.points) < 300:
        pcd_ds = copy.deepcopy(pcd)

    pcd_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30)
    )
    pcd_ds.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 1000.0]))
    pcd_ds.normalize_normals()
    return np.asarray(pcd_ds.points), np.asarray(pcd_ds.normals), extent


def generate_trial(points, extent, rng, max_rot_deg, trans_ratio, noise_ratio):
    R = random_rotation(max_rot_deg, rng)
    t = rng.uniform(-trans_ratio, trans_ratio, size=3) * extent
    T_gt = np.eye(4)
    T_gt[:3, :3] = R
    T_gt[:3, 3] = t

    source = transform_points(points, T_gt)
    if noise_ratio > 0:
        source = source + rng.normal(0.0, noise_ratio * extent, size=source.shape)
    return source, T_gt


def resolve_dataset_paths(dataset_args, data_dir):
    if dataset_args:
        out = []
        for item in dataset_args:
            if os.path.isabs(item) and os.path.exists(item):
                out.append(item)
                continue
            p = os.path.join(data_dir, item)
            if os.path.exists(p):
                out.append(p)
                continue
            stem = os.path.splitext(item)[0]
            found = None
            for ext in [".ply", ".obj", ".stl", ".off", ".pcd"]:
                cand = os.path.join(data_dir, stem + ext)
                if os.path.exists(cand):
                    found = cand
                    break
            if found is None:
                raise FileNotFoundError(f"Dataset not found: {item}")
            out.append(found)
        return out

    files = []
    for name in os.listdir(data_dir):
        lower = name.lower()
        if lower.endswith((".ply", ".obj", ".stl", ".off", ".pcd")):
            files.append(os.path.join(data_dir, name))
    if not files:
        raise FileNotFoundError(f"No dataset files found in: {data_dir}")
    return sorted(files)


def summarize(rows):
    grouped = {}
    for r in rows:
        k = (r["dataset"], r["mode"])
        grouped.setdefault(k, []).append(r)

    summary = []
    for (dataset, mode), items in grouped.items():
        arr_final = np.array([x["final_error"] for x in items], dtype=float)
        arr_rot = np.array([x["rot_err_deg"] for x in items], dtype=float)
        arr_trans = np.array([x["trans_err"] for x in items], dtype=float)
        arr_time = np.array([x["time_sec"] for x in items], dtype=float)
        arr_iter = np.array([x["iterations"] for x in items], dtype=float)
        summary.append({
            "dataset": dataset,
            "mode": mode,
            "trials": len(items),
            "final_error_mean": float(arr_final.mean()),
            "final_error_std": float(arr_final.std()),
            "rot_err_deg_mean": float(arr_rot.mean()),
            "rot_err_deg_std": float(arr_rot.std()),
            "trans_err_mean": float(arr_trans.mean()),
            "trans_err_std": float(arr_trans.std()),
            "time_sec_mean": float(arr_time.mean()),
            "iterations_mean": float(arr_iter.mean()),
        })
    return sorted(summary, key=lambda x: (x["dataset"], x["final_error_mean"]))


def main():
    parser = argparse.ArgumentParser(description="Benchmark ICP regularity across datasets and modes.")
    parser.add_argument("--data-dir", default=".", help="Folder containing dataset files.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset filenames or stems (e.g. rabbit armadillo).")
    parser.add_argument("--modes", nargs="+", default=["baseline", "weighted", "point_to_plane", "multires"])
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rot-deg", type=float, default=20.0)
    parser.add_argument("--trans-ratio", type=float, default=0.05)
    parser.add_argument("--noise-ratio", type=float, default=0.001)
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    datasets = resolve_dataset_paths(args.datasets, args.data_dir)

    rows = []
    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        target, target_normals, extent = load_dataset_points(ds_path)
        for mode in args.modes:
            for trial in range(args.trials):
                source, T_gt = generate_trial(
                    target, extent, rng, args.max_rot_deg, args.trans_ratio, args.noise_ratio
                )
                start = time.perf_counter()
                T_est, _, errs = run_icp_mode(source, target, mode=mode, target_normals=target_normals)
                elapsed = time.perf_counter() - start

                T_true = np.linalg.inv(T_gt)
                r_err = rotation_error_deg(T_est[:3, :3], T_true[:3, :3])
                t_err = float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))
                final_err = float(errs[-1]) if errs else float("nan")

                rows.append({
                    "dataset": ds_name,
                    "mode": mode,
                    "trial": trial,
                    "iterations": len(errs),
                    "final_error": final_err,
                    "rot_err_deg": r_err,
                    "trans_err": t_err,
                    "time_sec": elapsed,
                })

    raw_csv = os.path.join(args.output_dir, "benchmark_raw.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "mode", "trial", "iterations", "final_error", "rot_err_deg", "trans_err", "time_sec"]
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_csv = os.path.join(args.output_dir, "benchmark_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "mode", "trials", "final_error_mean", "final_error_std",
                "rot_err_deg_mean", "rot_err_deg_std", "trans_err_mean", "trans_err_std",
                "time_sec_mean", "iterations_mean"
            ]
        )
        writer.writeheader()
        writer.writerows(summary)

    txt = os.path.join(args.output_dir, "leaderboard.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Lower is better for error metrics.\n")
        for s in summary:
            line = (
                f"{s['dataset']:<20} {s['mode']:<16} "
                f"final={s['final_error_mean']:.6f}+/-{s['final_error_std']:.6f} "
                f"rot={s['rot_err_deg_mean']:.3f}deg "
                f"trans={s['trans_err_mean']:.6f} "
                f"time={s['time_sec_mean']:.4f}s\n"
            )
            f.write(line)

    print(f"Saved raw results: {raw_csv}")
    print(f"Saved summary:     {summary_csv}")
    print(f"Saved leaderboard: {txt}")


if __name__ == "__main__":
    main()
