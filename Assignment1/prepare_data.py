import numpy as np
import open3d as o3d
import copy

def prepare_icp_data(ply_path, output_npz_path):
    print(f"1. 正在加载原始模型: {ply_path}")
    # 斯坦福模型通常是三角网格形式，我们先读取为 Mesh 方便计算初始法线
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals

    # 2. 降采样 (极大加快 CPU 上的 KD-Tree 搜索速度)
    # 获取模型的包围盒大小，自适应计算 voxel_size
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    voxel_size = extent * 0.02  # 设置为对角线长度的 2%，大约能把点数降到 5000-10000 左右

    print(f"2. 执行体素降采样 (Voxel Size: {voxel_size:.4f})...")
    target_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 降采样后重新估算 Target 的法向量 (Point-to-plane 强依赖法线)
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=30)
    )
    # 统一法线朝向，防止正负号翻转影响点到面的计算
    target_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 1000.]))
    print(f"   Target 点云数量从 {len(pcd.points)} 降至 {len(target_pcd.points)}")

    # 3. 生成 Source 点云并应用 Ground Truth 变换
    source_pcd = copy.deepcopy(target_pcd)

    # 设定已知的旋转 (绕 Y 轴转 20 度) 和平移
    theta = np.radians(20)
    R = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [             0, 1,             0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    # 根据模型的实际尺寸设置平移量，确保 source 和 target 有一定错位但又不会完全分离
    t = np.array([extent * 0.05, extent * 0.02, -extent * 0.03])

    gt_transform = np.eye(4)
    gt_transform[:3, :3] = R
    gt_transform[:3, 3] = t

    # 变换 Source
    source_pcd.transform(gt_transform)
    source_points = np.asarray(source_pcd.points)

    # (可选) 加入极少量的高斯噪声，模拟真实世界传感器的测量误差，让作业看起来更真实
    noise_sigma = extent * 0.001
    source_points += np.random.normal(0, noise_sigma, source_points.shape)

    # 4. 提取为 Numpy 数组并保存
    target_points = np.asarray(target_pcd.points)
    target_normals = np.asarray(target_pcd.normals)

    np.savez(
        output_npz_path,
        source=source_points,
        target=target_points,
        target_normals=target_normals,
        ground_truth=gt_transform  # 保存真实矩阵，方便作业算误差画图
    )

    print(f"3. 数据已成功保存至: {output_npz_path}")
    print("-" * 40)
    print("NPZ 文件包含以下 Keys (在你的 icp_registration.py 中读取):")
    print(f" - data['source']         : {source_points.shape}")
    print(f" - data['target']         : {target_points.shape}")
    print(f" - data['target_normals'] : {target_normals.shape}  <-- 用于 Bonus 1 (Point-to-plane)")
    print(f" - data['ground_truth']   : (4, 4) 矩阵           <-- 用于画 convergence_plot.png")

if __name__ == "__main__":
    # 请确保同级目录下有你下载的 armadillo.ply
    prepare_icp_data("armadillo.ply", "test_data.npz")