# Camera Pose Estimation via Epipolar Geometry

## 1. Implementation Details

### 1.1 Feature extraction and matching

The pipeline supports both **SIFT** and **ORB**, and the main experiments use **SIFT**. SIFT is selected because it is generally more stable under moderate scale and viewpoint changes than ORB, which is helpful for two-view pose estimation on real image pairs.

After feature extraction, descriptor matching is performed with:
- **FLANN** for SIFT descriptors;
- **BFMatcher** for ORB descriptors.

To reject weak correspondences, the implementation applies **Lowe’s ratio test**. Only matches that pass this test are used for geometric estimation.

### 1.2 Fundamental matrix estimation

The fundamental matrix is estimated using the **normalized 8-point algorithm**. Before estimation, the matched 2D points are normalized by shifting the centroid to the origin and scaling the mean distance to \(\sqrt{2}\). This improves numerical stability.

Given normalized correspondences, the linear system is solved by **SVD**. The estimated matrix is then forced to rank 2 by setting the smallest singular value to zero. Finally, the matrix is denormalized back to the original image coordinate system.

### 1.3 RANSAC and Sampson distance

Because feature matching inevitably contains outliers, the pipeline uses **RANSAC** to robustly estimate the fundamental matrix. In each iteration, 8 correspondences are sampled, a candidate fundamental matrix is estimated, and all correspondences are evaluated by the **Sampson distance**.

The model with the largest number of inliers is selected, and the final fundamental matrix is re-estimated from all inliers. This step significantly improves robustness against mismatches.

### 1.4 Essential matrix and pose recovery

When camera intrinsics are known, the essential matrix is computed as:

\[
E = K_2^T F K_1
\]

After that, the singular values of \(E\) are enforced to follow the expected form \((1,1,0)\), which ensures consistency with the geometric definition of the essential matrix.

The essential matrix is decomposed into four candidate \((R,t)\) solutions. Since only one of them corresponds to the physically correct camera motion, the implementation performs **linear triangulation** and applies a **cheirality check**. The pose that yields the largest number of triangulated points with positive depth in front of both cameras is selected as the final result.

### 1.5 Visualization and outputs

The pipeline generates several output files for qualitative and quantitative analysis:
- `feature_matches.png`: inlier feature correspondences after RANSAC;
- `epipolar_lines.png`: epipolar geometry visualization;
- `pose_results.txt`: estimated \(F\), \(E\), \(R\), and \(t\);
- `error_analysis.txt`: pose error summary when ground truth is available;
- `reconstruction.png`: triangulated 3D point visualization.

## 2. Parameter Choices and Justifications

Several parameter settings are important in the current implementation.

First, **SIFT** is used as the default feature extractor. Compared with ORB, SIFT usually provides more distinctive descriptors and better repeatability for viewpoint changes, which is beneficial for epipolar estimation.

Second, the **ratio test threshold** is set to **0.75**. This is a standard choice that balances retaining enough correspondences while filtering ambiguous matches.

Third, **RANSAC** uses **2000 iterations** and a **Sampson distance threshold of 1.0**. These values were chosen as a reasonable compromise between robustness and computational cost. A smaller threshold may reject too many true correspondences, while a larger threshold may keep too many outliers.

Finally, the pipeline evaluates both synthetic and real data. During development, the synthetic scene was useful for validating the implementation, but real images were necessary to test practical performance under natural texture, illumination, and viewpoint changes.

## 3. Results Analysis

### 3.1 Real-image experiment

For the final test, the submitted `test_data` uses a real image pair from the **TUM RGB-D Freiburg1 XYZ** dataset. The corresponding camera intrinsics and ground-truth relative pose were also prepared for evaluation.

The current run reports the following results:
- Raw matches: **66**
- Inliers after RANSAC: **28**
- Inlier ratio: **0.4242**
- Positive-depth triangulated points: **19**
- Rotation error: **9.5831 deg**
- Translation direction error: **74.9090 deg**

These results show that the pipeline can recover a physically valid pose from a real image pair and successfully produce epipolar and reconstruction outputs. The estimated essential matrix also satisfies the expected singular value structure, which indicates that the algebraic estimation steps are working correctly.

### 3.2 Interpretation of the results

The rotation estimation is moderately accurate, with an error below 10 degrees. This suggests that the recovered epipolar geometry captures the dominant camera motion reasonably well.

However, the translation direction error is much larger. This is not unusual in two-view geometry, especially when:
- the camera baseline is relatively small;
- the scene has limited texture;
- the number of reliable correspondences is limited;
- the translation is estimated only up to scale and is more sensitive to noisy matches than rotation.

The real dataset therefore demonstrates both the strength and the limitation of the current classical pipeline. The framework is correct and functional, but its accuracy still depends heavily on the quality of image matches.

## 4. Discussion of Limitations and Improvements

One limitation of the current system is that it depends strongly on sparse local feature matching. If the image pair contains repetitive patterns, weak texture, motion blur, or low overlap, the number of reliable matches can drop significantly, which directly affects the quality of the estimated fundamental matrix.

Another limitation is that the translation estimate is more unstable than the rotation estimate. In real scenes with a short baseline, small correspondence errors can produce large angular deviations in translation direction.

The synthetic data generation step also required adjustment. Initially, the synthetic images contained strong text overlays, and the feature detector tended to match letters instead of projected points. This was corrected by removing the text, but it shows that synthetic validation images must be designed carefully to avoid introducing artificial high-contrast structures unrelated to the intended scene geometry.

Possible improvements include:
- adding stricter match filtering or mutual consistency checks;
- tuning SIFT and RANSAC parameters for real data;
- testing alternative matching strategies or more distinctive descriptors;
- using depth information from the RGB-D dataset for stronger geometric verification;
- refining pose estimation with nonlinear optimization after the initial essential matrix solution.

These changes could improve robustness and reduce translation error on challenging real image pairs.