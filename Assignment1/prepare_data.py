import numpy as np
import open3d as o3d
import copy

def prepare_icp_data(ply_path, output_npz_path):
    print(f"1. Loading original model: {ply_path}")
    # The Stanford model is usually a triangle mesh, so we read it as a Mesh first to easily compute initial normals
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals

    # 2. Downsampling (Significantly speeds up KD-Tree search on the CPU)
    # Get the bounding box of the model to adaptively calculate voxel_size
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    voxel_size = extent * 0.02  # Set to 2% of the diagonal length, reducing the point count to roughly 5000-10000

    print(f"2. Performing voxel downsampling (Voxel Size: {voxel_size:.4f})...")
    target_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Re-estimate normals of the Target after downsampling (Point-to-plane relies heavily on normals)
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=30)
    )
    # Orient normals consistently to prevent sign flips from affecting point-to-plane calculations
    target_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 1000.]))
    print(f"   Target point cloud size reduced from {len(pcd.points)} to {len(target_pcd.points)}")

    # 3. Generate Source point cloud and apply Ground Truth transformation
    source_pcd = copy.deepcopy(target_pcd)

    # Set known rotation (20 degrees around the Y-axis) and translation
    theta = np.radians(20)
    R = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [             0, 1,             0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    # Set translation based on the actual size of the model, ensuring source and target are misaligned but not completely separated
    t = np.array([extent * 0.05, extent * 0.02, -extent * 0.03])

    gt_transform = np.eye(4)
    gt_transform[:3, :3] = R
    gt_transform[:3, 3] = t

    # Transform Source
    source_pcd.transform(gt_transform)
    source_points = np.asarray(source_pcd.points)

    # (Optional) Add a tiny amount of Gaussian noise to simulate real-world sensor measurement errors and make the assignment look more realistic
    noise_sigma = extent * 0.001
    source_points += np.random.normal(0, noise_sigma, source_points.shape)

    # 4. Extract as Numpy arrays and save
    target_points = np.asarray(target_pcd.points)
    target_normals = np.asarray(target_pcd.normals)

    np.savez(
        output_npz_path,
        source=source_points,
        target=target_points,
        target_normals=target_normals,
        ground_truth=gt_transform  # Save the ground truth matrix for calculating errors and plotting in the assignment
    )

    print(f"3. Data successfully saved to: {output_npz_path}")
    print("-" * 40)
    print("The NPZ file contains the following Keys (to be read in your icp_registration.py):")
    print(f" - data['source']         : {source_points.shape}")
    print(f" - data['target']         : {target_points.shape}")
    print(f" - data['target_normals'] : {target_normals.shape}  <-- Used for Bonus 1 (Point-to-plane)")
    print(f" - data['ground_truth']   : (4, 4) Matrix         <-- Used for plotting convergence_plot.png")

if __name__ == "__main__":
    # Please make sure the downloaded armadillo.ply is in the same directory
    prepare_icp_data("armadillo.ply", "test_data.npz")