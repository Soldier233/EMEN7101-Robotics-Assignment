import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
import time
import open3d as o3d
try:
    from prepare_data import prepare_icp_data
except ModuleNotFoundError:
    from Assignment1.prepare_data import prepare_icp_data

def save_ply(filename, points):
    """Save point cloud to a .ply file."""
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
end_header
""".format(len(points))

    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt='%.6f %.6f %.6f')


def best_fit_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def rodrigues(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.identity(3) + skew(omega)
    k = omega / theta
    K = skew(k)
    return np.identity(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def estimate_normals(points, radius=0.03, max_nn=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 1000.0]))
    pcd.normalize_normals()
    return np.asarray(pcd.normals)


def downsample_points(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd_ds.points)


def compute_weights(distances, src=None, tgt_normals=None, mode='distance', sigma=None):
    d = distances.ravel()
    if sigma is None:
        sigma = max(float(np.median(d)), 1e-6)
    if mode == 'distance':
        w = np.exp(-(d ** 2) / (2.0 * sigma * sigma))
    elif mode == 'normal' and src is not None and tgt_normals is not None:
        src_normals = estimate_normals(src)
        cosine = np.abs(np.sum(src_normals * tgt_normals, axis=1))
        w = np.clip(cosine, 1e-3, 1.0)
    elif mode == 'both' and src is not None and tgt_normals is not None:
        src_normals = estimate_normals(src)
        cosine = np.abs(np.sum(src_normals * tgt_normals, axis=1))
        w_dist = np.exp(-(d ** 2) / (2.0 * sigma * sigma))
        w = np.clip(cosine, 1e-3, 1.0) * w_dist
    else:
        w = np.ones_like(d)
    return np.clip(w, 1e-6, None)


def best_fit_transform_weighted(A, B, weights):
    w = weights.reshape(-1, 1)
    w_sum = np.sum(w)
    centroid_A = np.sum(w * A, axis=0) / w_sum
    centroid_B = np.sum(w * B, axis=0) / w_sum

    AA = A - centroid_A
    BB = B - centroid_B
    H = (w * AA).T @ BB

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def point_to_plane_step(src, tgt, tgt_normals, weights=None):
    if weights is None:
        weights = np.ones(src.shape[0], dtype=float)

    A = np.zeros((src.shape[0], 6), dtype=float)
    b = np.zeros(src.shape[0], dtype=float)

    cross_terms = np.cross(src, tgt_normals)
    A[:, :3] = cross_terms
    A[:, 3:] = tgt_normals
    b[:] = np.sum(tgt_normals * (tgt - src), axis=1)

    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    Aw = A * sqrt_w
    bw = b * sqrt_w.ravel()
    x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    omega = x[:3]
    t = x[3:]
    R = rodrigues(omega)

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def robust_plane_weights(src, tgt, normals, c=2.5):
    residual = np.sum(normals * (tgt - src), axis=1)
    scale = 1.4826 * np.median(np.abs(residual)) + 1e-9
    u = residual / (c * scale)
    return 1.0 / (1.0 + u * u)


def icp(source, target, max_iterations=50, tolerance=1e-5, method='point_to_point',
        use_weights=False, weight_mode='distance', target_normals=None,
        trim_quantile=1.0, max_corr_dist=None, robust_plane=False):
    src = np.copy(source)
    T_final = np.identity(4)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)

    if method == 'point_to_plane' and target_normals is None:
        target_normals = estimate_normals(target)

    prev_error = float('inf')
    error_history = []

    for _ in range(max_iterations):
        distances, indices = nbrs.kneighbors(src)
        d = distances.ravel()
        mask = np.ones_like(d, dtype=bool)
        if max_corr_dist is not None:
            mask &= (d <= max_corr_dist)
        if trim_quantile < 1.0:
            cutoff = np.quantile(d, trim_quantile)
            mask &= (d <= cutoff)
        if np.count_nonzero(mask) < 20:
            break

        current_error = float(np.mean(d[mask]))
        error_history.append(current_error)

        if np.abs(prev_error - current_error) < tolerance:
            break
        prev_error = current_error

        idx = indices.ravel()[mask]
        src_in = src[mask]
        d_in = d[mask]
        target_matched = target[idx]
        matched_normals = target_normals[idx] if target_normals is not None else None

        weights = np.ones(src_in.shape[0], dtype=float)
        if use_weights:
            weights = compute_weights(
                d_in.reshape(-1, 1), src=src_in, tgt_normals=matched_normals, mode=weight_mode
            )

        if method == 'point_to_plane':
            if robust_plane:
                weights = weights * robust_plane_weights(src_in, target_matched, matched_normals)
            T = point_to_plane_step(src_in, target_matched, matched_normals, weights=weights)
        else:
            T = best_fit_transform_weighted(src_in, target_matched, weights)

        src = src @ T[:3, :3].T + T[:3, 3]
        T_final = T @ T_final

    return T_final, src, error_history


def multi_resolution_icp(source, target, voxel_sizes=(0.08, 0.04, 0.02),
                         max_iterations=(40, 30, 20), tolerance=1e-5,
                         method='point_to_plane', use_weights=True, weight_mode='distance'):
    T_total = np.identity(4)
    error_history = []

    for voxel_size, iters in zip(voxel_sizes, max_iterations):
        src_level = source @ T_total[:3, :3].T + T_total[:3, 3]
        src_ds = downsample_points(src_level, voxel_size)
        tgt_ds = downsample_points(target, voxel_size)
        tgt_normals = estimate_normals(tgt_ds, radius=voxel_size * 2.5, max_nn=30)

        T_level, _, errors = icp(
            src_ds, tgt_ds, max_iterations=iters, tolerance=tolerance,
            method=method, use_weights=use_weights, weight_mode=weight_mode,
            target_normals=tgt_normals
        )
        T_total = T_level @ T_total
        error_history.extend(errors)

    aligned = source @ T_total[:3, :3].T + T_total[:3, 3]
    return T_total, aligned, error_history


def run_icp_mode(source, target, mode="baseline", target_normals=None):
    if mode == "baseline":
        return icp(source, target, max_iterations=50, tolerance=1e-5,
                   method='point_to_point', use_weights=False)
    if mode == "weighted":
        return icp(source, target, max_iterations=50, tolerance=1e-5,
                   method='point_to_point', use_weights=True, weight_mode='distance')
    if mode == "point_to_plane":
        # Warm start with point-to-point, then robust point-to-plane refinement.
        T0, src0, e0 = icp(source, target, max_iterations=20, tolerance=1e-5,
                           method='point_to_point', use_weights=True, weight_mode='distance',
                           trim_quantile=0.9)
        T1, src1, e1 = icp(src0, target, max_iterations=35, tolerance=1e-6,
                           method='point_to_plane', use_weights=True, weight_mode='distance',
                           trim_quantile=0.85, robust_plane=True, target_normals=target_normals)
        return T1 @ T0, src1, e0 + e1
    if mode == "point_to_plane_raw":
        return icp(source, target, max_iterations=50, tolerance=1e-5,
                   method='point_to_plane', use_weights=False, target_normals=target_normals)
    if mode == "multires":
        return multi_resolution_icp(
            source, target,
            voxel_sizes=(0.08, 0.04, 0.02),
            max_iterations=(35, 25, 15),
            method='point_to_plane',
            use_weights=True,
            weight_mode='distance'
        )
    raise ValueError(f"Unknown ICP mode: {mode}")


def benchmark_icp_variants(source, target, output_path):
    variants = [
        ("point_to_point", dict(method='point_to_point', use_weights=False)),
        ("weighted_point_to_point", dict(method='point_to_point', use_weights=True, weight_mode='distance')),
        ("point_to_plane_raw", dict(method='point_to_plane', use_weights=False)),
        ("weighted_point_to_plane_robust", dict(method='point_to_plane', use_weights=True, weight_mode='distance', trim_quantile=0.85, robust_plane=True)),
    ]

    lines = ["variant,iterations,final_error,time_sec"]
    for name, kwargs in variants:
        start = time.perf_counter()
        _, _, errs = icp(source, target, max_iterations=50, tolerance=1e-5, **kwargs)
        elapsed = time.perf_counter() - start
        final_err = errs[-1] if errs else float('nan')
        lines.append(f"{name},{len(errs)},{final_err:.8f},{elapsed:.4f}")

    start = time.perf_counter()
    _, _, errs = multi_resolution_icp(
        source, target, voxel_sizes=(0.08, 0.04, 0.02), max_iterations=(35, 25, 15),
        method='point_to_plane', use_weights=True, weight_mode='distance'
    )
    elapsed = time.perf_counter() - start
    final_err = errs[-1] if errs else float('nan')
    lines.append(f"multi_resolution_weighted_point_to_plane,{len(errs)},{final_err:.8f},{elapsed:.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    # 1. Prepare data
    if not os.path.exists('test_data.npz'):
        if os.path.exists("armadillo.ply"):
            ply_path = "armadillo.ply"
        elif os.path.exists("Armadillo.ply"):
            ply_path = "Armadillo.ply"
        else:
            raise FileNotFoundError("Could not find armadillo.ply/Armadillo.ply in current directory.")
        prepare_icp_data(ply_path, "test_data.npz")

    data = np.load('test_data.npz')
    source = data['source']
    target = data['target']
    target_normals = data['target_normals'] if 'target_normals' in data.files else None

    # 2. Run ICP
    # Modes: baseline | weighted | point_to_plane | point_to_plane_raw | multires
    icp_mode = os.getenv("ICP_MODE", "baseline")
    print(f"Running ICP mode: {icp_mode}")
    T_matrix, aligned_source, errors = run_icp_mode(
        source, target, mode=icp_mode, target_normals=target_normals
    )

    # 3. Create results directory
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to '{output_dir}/' ...")

    # --- A. Save transformation.txt ---
    np.savetxt(os.path.join(output_dir, 'transformation.txt'), T_matrix,
               fmt='%.8f', header='ICP Transformation Matrix (4x4)')

    # --- B. Save aligned.ply ---
    save_ply(os.path.join(output_dir, 'aligned.ply'), aligned_source)
    # Also save the target for comparison (optional)
    save_ply(os.path.join(output_dir, 'target_ref.ply'), target)

    # --- C. Save convergence_curve.png ---
    plt.figure(figsize=(8, 4))
    plt.plot(errors, marker='o', linestyle='-', color='b')
    plt.title('ICP Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Euclidean Distance Error')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'convergence_curve.png'))
    plt.close()

    # --- D. Save visualization.png ---
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Original Target (Blue)
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='blue', s=1, alpha=0.5, label='Target')
    # Aligned Source (Red)
    ax.scatter(aligned_source[:, 0], aligned_source[:, 1], aligned_source[:, 2], c='red', s=1, alpha=0.5, label='Aligned Source')

    ax.set_title('ICP Alignment Result')
    ax.legend()
    # Fix the view angle for consistent screenshots
    ax.view_init(elev=30, azim=45)
    plt.savefig(os.path.join(output_dir, 'visualization.png'), dpi=150)
    plt.close()

    # --- E. Optional benchmarking output ---
    benchmark_icp_variants(source, target, os.path.join(output_dir, 'benchmark.csv'))

    print("All offline tasks completed!")

    # --- F. Launch Open3D interactive visualization window ---
    print("Launching Open3D visualization window (Left click to drag/rotate, right click to pan, scroll to zoom)...")

    # 1. Create Target point cloud object and paint color
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target)
    pcd_target.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

    # 2. Create Aligned Source point cloud object and paint color
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(aligned_source)
    pcd_aligned.paint_uniform_color([1.0, 0.0, 0.0])  # Red

    # 3. Create Original Source point cloud object (optional, set to green for comparison)
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)
    pcd_source.paint_uniform_color([0.0, 1.0, 0.0])  # Green

    # Display window (the program will pause until you close this popup window)
    o3d.visualization.draw_geometries(
        [pcd_target, pcd_aligned, pcd_source],
        window_name="Interactive ICP Result Visualization (Blue: Target, Red: Aligned, Green: Source)",
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False
    )

    print("Visualization window closed. Program finished.")
