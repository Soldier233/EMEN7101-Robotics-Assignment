import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

# ==========================================
# 1. 辅助功能：数据生成与文件保存
# ==========================================

def generate_torus_point_cloud(n_samples=1000, R=3.0, r=1.0):
    """生成一个圆环点云作为示例数据"""
    theta = 2 * np.pi * np.random.rand(n_samples)
    phi = 2 * np.pi * np.random.rand(n_samples)

    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    return np.vstack((x, y, z)).T

def save_ply(filename, points):
    """将点云保存为 .ply 格式"""
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

def create_sample_npz(filename='sample_data.npz'):
    """生成并保存示例数据到 .npz"""
    print(f"正在生成示例数据: {filename} ...")
    # 生成目标点云 (Target)
    target = generate_torus_point_cloud(1000)

    # 生成源点云 (Source): 旋转 + 平移 + 噪声
    theta = np.radians(45)
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    t_true = np.array([2.0, 3.0, -1.0])

    source = np.dot(target, R_true.T) + t_true
    source += np.random.normal(0, 0.05, size=source.shape) # 添加噪声

    # 保存为 npz
    np.savez(filename, source=source, target=target)
    print("数据生成完毕。")

# ==========================================
# 2. 核心 ICP 算法 (带误差记录)
# ==========================================

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
    error_history = [] # 用于记录收敛曲线

    for i in range(max_iterations):
        distances, indices = nbrs.kneighbors(src)
        current_error = np.mean(distances)
        error_history.append(current_error)

        # 检查收敛
        if np.abs(prev_error - current_error) < tolerance:
            break
        prev_error = current_error

        target_matched = target[indices.ravel()]
        T = best_fit_transform(src, target_matched)

        # 更新源点云和变换矩阵
        src = np.dot(src, T[:3, :3].T) + T[:3, 3]
        T_final = np.dot(T, T_final)

    return T_final, src, error_history

# ==========================================
# 3. 主程序：执行并导出结果
# ==========================================

if __name__ == "__main__":
    # 1. 准备数据
    if not os.path.exists('sample_data.npz'):
        create_sample_npz()

    data = np.load('sample_data.npz')
    source = data['source']
    target = data['target']

    # 2. 运行 ICP
    print("正在运行 ICP 算法...")
    T_matrix, aligned_source, errors = icp(source, target)

    # 3. 创建结果目录
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在保存结果到 '{output_dir}/' ...")

    # --- A. 保存 transformation.txt ---
    np.savetxt(os.path.join(output_dir, 'transformation.txt'), T_matrix,
               fmt='%.8f', header='ICP Transformation Matrix (4x4)')

    # --- B. 保存 aligned.ply ---
    save_ply(os.path.join(output_dir, 'aligned.ply'), aligned_source)
    # 顺便也保存一下 target 以便对比 (可选)
    save_ply(os.path.join(output_dir, 'target_ref.ply'), target)

    # --- C. 保存 收敛曲线.png ---
    plt.figure(figsize=(8, 4))
    plt.plot(errors, marker='o', linestyle='-', color='b')
    plt.title('ICP Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Euclidean Distance Error')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'convergence_curve.png'))
    plt.close()

    # --- D. 保存 可视化.png ---
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    # 原始 Target (蓝色)
    ax.scatter(target[:,0], target[:,1], target[:,2], c='blue', s=1, alpha=0.5, label='Target')
    # 对齐后的 Source (红色)
    ax.scatter(aligned_source[:,0], aligned_source[:,1], aligned_source[:,2], c='red', s=1, alpha=0.5, label='Aligned Source')

    ax.set_title('ICP Alignment Result')
    ax.legend()
    # 固定视角以便截图一致
    ax.view_init(elev=30, azim=45)
    plt.savefig(os.path.join(output_dir, 'visualization.png'), dpi=150)
    plt.close()

    print("所有任务完成！请查看 results 文件夹。")