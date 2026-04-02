# Visual Bag of Words for Image Retrieval

## 1. Objective

This assignment implements a complete Visual Bag of Words (BoW) pipeline for image retrieval. The system extracts local image descriptors, clusters them into a visual vocabulary, converts each image into a BoW histogram with TF-IDF weighting, and ranks reference images for each query by histogram similarity. A lightweight loop-closure module is also included to satisfy the bonus direction described in the assignment brief.

The implementation is organized into the following files:

- `bow_retrieval.py`: feature extraction, vocabulary construction, BoW encoding, similarity computation, image retrieval, evaluation, and optional loop closure detection.
- `run_retrieval.py`: command-line entry point that can also generate a deterministic demo dataset when no external dataset is available.
- `config.yaml`: experiment configuration, including feature type, vocabulary size, retrieval metric, and output paths.
- `results/`: generated outputs including the vocabulary, TF-IDF weights, retrieval rankings, metrics, loop-closure predictions, and retrieval visualizations.

## 2. Method

### 2.1 Feature Extraction

Each image is processed with OpenCV and converted to grayscale. I used **SIFT** as the main local feature extractor because it provides scale- and rotation-invariant descriptors and is usually more stable than ORB for retrieval on moderate viewpoint changes. The function `extract_features` returns both OpenCV keypoints and a descriptor matrix with shape `N x D`.

If an image does not produce any descriptor, the pipeline keeps an empty descriptor array rather than failing. This makes the code robust to low-texture or corrupted inputs.

### 2.2 Vocabulary Construction

All descriptors from the reference database are pooled together and clustered by **MiniBatchKMeans**. The cluster centers become the visual words. Compared with standard K-means, MiniBatchKMeans is computationally lighter and is better suited to larger descriptor sets while still producing a usable vocabulary for this assignment.

The vocabulary size was set to **48** in the provided configuration. This value is large enough to separate the demo scenes while remaining compact for a small assignment-scale dataset.

### 2.3 Image Representation

For each descriptor, the nearest visual word is found by Euclidean distance to the cluster centers. The image is then represented as a word-frequency histogram. To improve discriminative power, I applied **TF-IDF weighting**:

- term frequency (TF) is the normalized word count in the image;
- inverse document frequency (IDF) down-weights common visual words that appear in many database images.

The final BoW vector is L2-normalized before matching. This normalization works especially well with cosine similarity.

### 2.4 Similarity Matching

The implementation supports several histogram comparison metrics:

- cosine similarity;
- L1 distance converted to a similarity score;
- L2 distance converted to a similarity score;
- chi-squared distance converted to a similarity score.

The main experiment uses **cosine similarity**, which is a natural fit for normalized TF-IDF histograms.

To improve robustness, I also added an optional **spatial verification** stage as a bonus extension. After the initial BoW ranking, the top candidates are re-ranked using descriptor matching with Lowe's ratio test followed by RANSAC homography estimation. The number of inliers is used as a geometric consistency score. This keeps the BoW pipeline as the main retrieval backbone while reducing false positives among visually similar scenes.

### 2.5 Loop Closure Detection

As a bonus extension, the pipeline includes `detect_loop_closure`. Given a sequence of image histograms, the current frame is compared against earlier frames while skipping the most recent ones through a minimum temporal gap. A loop closure is reported when the best similarity score exceeds a threshold. This is a simple but interpretable baseline for place revisiting.

## 3. Experiment Setup

The repository originally did not include a real retrieval dataset, so I created a **deterministic synthetic demo dataset** to make the pipeline fully runnable and reproducible offline. The demo generator creates five scene classes (`alpha` to `epsilon`), each with:

- three reference images containing geometric structures and textured details;
- one query image generated from the same base scene with rotation, translation, blur, and noise.

This setup is intentionally small, but it allows quantitative validation of the full retrieval pipeline and produces the exact output files required by the assignment.

Key parameters from `config.yaml` are:

- feature extractor: SIFT;
- maximum local features per image: 900;
- vocabulary size: 48;
- retrieval metric: cosine similarity;
- spatial verification: enabled for the top 5 BoW candidates;
- top-k evaluation: `k = 3`.

## 4. Results

Running `python3 Assignment2/run_retrieval.py --generate-demo` generates the outputs in `Assignment2/results/`. On the included demo dataset, the retrieval system produced:

- **Top-1 accuracy: 1.000**
- **Top-3 accuracy: 1.000**
- **MRR: 1.000**

These results indicate that the synthetic queries are consistently matched to the correct reference scene at rank 1. This is expected because the query images preserve the core local structures of their corresponding scene while still introducing controlled appearance changes. The additional spatial verification stage is especially helpful for resolving ambiguous candidates that have similar BoW histograms but inconsistent local geometry.

The saved visualizations confirm that the top-ranked matches remain visually consistent with the query even when the query is rotated or slightly blurred. In addition, the loop-closure module produces a small set of detections on the query stream, demonstrating how the same BoW representation can be reused for place recognition.

## 5. Discussion

### Advantages

- The pipeline is modular and easy to extend.
- SIFT + TF-IDF + cosine similarity provides a strong classical baseline.
- MiniBatchKMeans keeps vocabulary construction efficient.
- RANSAC-based re-ranking improves precision without changing the underlying BoW representation.
- The implementation exports all major intermediate results for inspection and grading.

### Limitations

- Hard assignment maps each descriptor to only one visual word, which can lose information near cluster boundaries.
- Pure BoW discards geometric layout, so false positives are still possible before the optional verification stage.
- The current evaluation uses a synthetic dataset rather than a large real-world benchmark.
- Loop closure detection is based only on histogram similarity and does not include geometric verification.

## 6. Possible Improvements

Several extensions would improve retrieval quality beyond the baseline:

- soft assignment of descriptors to multiple nearby words;
- spatial verification with RANSAC after BoW ranking;
- inverted indexing for faster large-scale search;
- hierarchical vocabularies or vocabulary trees;
- VLAD or Fisher Vector aggregation for stronger global representations.

## 7. Conclusion

This assignment successfully implements a complete classical image retrieval pipeline based on the Visual Bag of Words model. The code covers all core steps requested in the brief, produces reproducible outputs, and includes an English report and example results. Although the method is simple compared with modern deep retrieval approaches, it remains an effective and interpretable baseline for image retrieval and loop closure detection.
