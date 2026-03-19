# Report On ICP Registration

## 1. Objective

This assignment implements and evaluates several variants of the Iterative Closest Point (ICP) algorithm for rigid 3D point-cloud registration. The pipeline first generates a synthetic registration task from the Stanford Armadillo model, then compares multiple ICP strategies in terms of alignment accuracy, robustness, and runtime.

The implementation is mainly organized in:
- `prepare_data.py`: builds source/target point clouds and ground-truth transformation.
- `icp_registration.py`: implements ICP variants and visualization/output generation.
- `benchmark_icp.py`: runs repeated trials and summarizes quantitative performance.

## 2. Method

### 2.1 Data preparation

The dataset is generated from `Armadillo.ply`. In `prepare_data.py`, the mesh is converted to a point cloud, voxel-downsampled, and its normals are estimated. A synthetic source cloud is then created by applying a known rigid transformation and small Gaussian noise to the target cloud. This provides a controlled benchmark with known geometric correspondence.

### 2.2 ICP variants

The code evaluates four main modes in `benchmark_icp.py`:
- **baseline**: standard point-to-point ICP using nearest-neighbor correspondences and SVD-based rigid transform estimation;
- **weighted**: point-to-point ICP with distance-based correspondence weighting;
- **point_to_plane**: a hybrid pipeline using a weighted point-to-point warm start followed by robust point-to-plane refinement;
- **multires**: multi-resolution point-to-plane ICP from coarse to fine voxel scales.

The point-to-plane version is the most advanced configuration because it uses surface normals and robust weighting, which should improve convergence when local geometry is informative.

## 3. Results

The benchmark summary in `benchmark_results/benchmark_summary.csv` reports the following averaged results over 8 trials on `Armadillo.ply`:

| Method | Mean RMSE | Rotation Error (deg) | Translation Error | Mean Time (s) | Mean Iterations |
|---|---:|---:|---:|---:|---:|
| point_to_plane | 0.3466 | 0.0551 | 0.0571 | 0.0552 | 31.25 |
| weighted | 0.3959 | 0.0255 | 0.0214 | 0.0244 | 15.88 |
| baseline | 0.3965 | 0.0229 | 0.0190 | 0.0179 | 8.63 |
| multires | 6.9500 | 4.2050 | 9.4918 | 0.1053 | 28.00 |

### 3.1 Accuracy

The **point-to-plane** method achieves the best RMSE, indicating the best final geometric alignment in this benchmark. This is consistent with the theory of ICP: point-to-plane updates often converge faster and more accurately near the correct pose because they minimize error along local surface normals instead of full Euclidean point distance.

The **baseline** and **weighted** point-to-point methods produce very similar RMSE values. The weighted version slightly improves RMSE, but the gain is small in the current setup.

The **multires** method performs much worse than expected, with very large RMSE and pose errors. This suggests that the current coarse-to-fine parameter choice is not well matched to this dataset or that downsampling at coarse levels loses important geometric detail.

### 3.2 Efficiency

The **baseline** method is the fastest and requires the fewest iterations. The **weighted** version is only slightly slower. The **point-to-plane** pipeline is slower because it includes both a warm-start stage and a robust refinement stage, but the runtime remains small. The **multires** method is the slowest among the tested approaches while also giving the weakest accuracy, so it is not advantageous in the present configuration.

## 4. Declaration

All the output of `results` folder is conducted using **baseline** method.
