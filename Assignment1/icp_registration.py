import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
import open3d as o3d
from prepare_data import prepare_icp_data

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


def icp(source, target, max_iterations=50, tolerance=1e-5):
    src = np.copy(source)
    T_final = np.identity(4)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)

    prev_error = float('inf')
    error_history = []  # To record the convergence curve

    for i in range(max_iterations):
        distances, indices = nbrs.kneighbors(src)
        current_error = np.mean(distances)
        error_history.append(current_error)

        # Check for convergence
        if np.abs(prev_error - current_error) < tolerance:
            break
        prev_error = current_error

        target_matched = target[indices.ravel()]
        T = best_fit_transform(src, target_matched)

        # Update source point cloud and transformation matrix
        src = np.dot(src, T[:3, :3].T) + T[:3, 3]
        T_final = np.dot(T, T_final)

    return T_final, src, error_history


if __name__ == "__main__":
    # 1. Prepare data
    if not os.path.exists('test_data.npz'):
        prepare_icp_data("armadillo.ply", "test_data.npz")

    data = np.load('test_data.npz')
    source = data['source']
    target = data['target']

    # 2. Run ICP
    print("Running ICP algorithm...")
    T_matrix, aligned_source, errors = icp(source, target)

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

    print("All offline tasks completed!")

    # --- E. Launch Open3D interactive visualization window ---
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