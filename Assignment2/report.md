# Visual Bag of Words for Image Retrieval

## 1. Objective

This assignment implements a complete Visual Bag of Words (BoW) pipeline for image retrieval. The system extracts local image descriptors, clusters them into a visual vocabulary, converts each image into a BoW histogram with TF-IDF weighting, and ranks reference images for each query by histogram similarity. A lightweight loop-closure module is also included to satisfy the bonus direction described in the assignment brief.

The implementation now supports two dataset modes:

- `demo`: a deterministic synthetic dataset that keeps the original assignment workflow runnable offline.
- `oxford5k`: the Oxford Buildings Dataset with automatic download, ground-truth parsing, query crop generation, and benchmark-style evaluation.

The implementation is organized into the following files:

- `bow_retrieval.py`: feature extraction, vocabulary construction, BoW encoding, similarity computation, image retrieval, dataset-aware evaluation, and optional loop closure detection.
- `run_retrieval.py`: command-line entry point, demo dataset generation, Oxford5k download/extraction, ground-truth parsing, and query crop preparation.
- `config.yaml`: experiment configuration, including dataset mode, feature type, vocabulary size, retrieval metric, evaluation settings, and output paths.
- `results/`: generated outputs including the vocabulary, TF-IDF weights, retrieval rankings, metrics, loop-closure predictions, and retrieval visualizations.

## 2. Method

### 2.1 Feature Extraction

Each image is processed with OpenCV and converted to grayscale. The default local feature extractor is **SIFT** because it provides scale- and rotation-invariant descriptors and is usually more stable than ORB for retrieval under moderate viewpoint changes. The function `extract_features` returns both OpenCV keypoints and a descriptor matrix with shape `N x D`.

If an image does not produce any descriptor, the pipeline keeps an empty descriptor array rather than failing. This is important for Oxford query crops because some crops may contain only limited texture.

### 2.2 Vocabulary Construction

All descriptors from the reference database are pooled together and clustered by **MiniBatchKMeans**. The cluster centers become the visual words. Compared with standard K-means, MiniBatchKMeans is computationally lighter and is better suited to larger descriptor sets while still producing a usable vocabulary for this assignment.

The vocabulary size and descriptor cap are configurable in `config.yaml`. This is especially important for Oxford5k because the real dataset is much larger than the synthetic demo set.

### 2.3 Image Representation

For each descriptor, the nearest visual word is found by Euclidean distance to the cluster centers. The image is then represented as a word-frequency histogram. To improve discriminative power, the implementation applies **TF-IDF weighting**:

- term frequency (TF) is the normalized word count in the image;
- inverse document frequency (IDF) down-weights common visual words that appear in many database images.

The final BoW vector is L2-normalized before matching. This normalization works especially well with cosine similarity.

### 2.4 Similarity Matching and Re-ranking

The implementation supports several histogram comparison metrics:

- cosine similarity;
- L1 distance converted to a similarity score;
- L2 distance converted to a similarity score;
- chi-squared distance converted to a similarity score.

The main experiment uses **cosine similarity**, which is a natural fit for normalized TF-IDF histograms.

To improve robustness, the pipeline also includes an optional **spatial verification** stage. After the initial BoW ranking, the top candidates are re-ranked using descriptor matching with Lowe's ratio test followed by RANSAC homography estimation. The number of inliers is used as a geometric consistency score.

### 2.5 Dataset Modes

#### Demo mode

The demo mode preserves the original assignment workflow. If the configured folders are empty and `auto_generate_if_missing` is enabled, the script generates five deterministic scene classes (`alpha` to `epsilon`) with reference and query images.

#### Oxford5k mode

The Oxford5k mode follows the benchmark protocol more closely:

- archives are downloaded automatically when missing;
- dataset files are extracted into fixed project directories;
- ground-truth files are parsed from the official Oxford format;
- each query crop is generated from the original source image and bounding box;
- evaluation is based on Oxford `good`, `ok`, and `junk` relevance rules.

Each Oxford query is represented with explicit metadata:

- `query_name`;
- `source_image_id`;
- source image path;
- crop bounding box;
- generated query image path;
- `relevant_ids = good ∪ ok`;
- `junk_ids`.

### 2.6 Loop Closure Detection

As a bonus extension, the pipeline includes `detect_loop_closure`. Given a sequence of query histograms, the current frame is compared against earlier frames while skipping the most recent ones through a minimum temporal gap. A loop closure is reported when the best similarity score exceeds a threshold. This is a simple but interpretable baseline for place revisiting.

## 3. Experiment Setup

Key configuration blocks in `config.yaml` are:

- `dataset.mode`: selects `demo` or `oxford5k`;
- `dataset.demo`: demo reference/query paths and auto-generation behavior;
- `dataset.oxford5k`: Oxford data root, image/GT/query-crop directories, archive directory, auto-download flag, and download URLs;
- `feature`: feature extractor type and maximum number of local features;
- `vocabulary`: vocabulary size, descriptor cap, and random seed;
- `retrieval`: ranking depth and similarity metric;
- `evaluation`: evaluation protocol and top-k values to report.

Example commands:

```bash
python Assignment2/run_retrieval.py --generate-demo
```

Runs the synthetic demo workflow.

```bash
python Assignment2/run_retrieval.py
```

Runs the mode currently selected in `Assignment2/config.yaml`. If `dataset.mode: oxford5k`, the script prepares Oxford5k automatically.

## 4. Evaluation Protocol

### 4.1 Demo mode

The demo dataset keeps the original label-based evaluation:

- Top-1 accuracy
- Top-k accuracy
- MRR

Correctness is determined by whether the first matching scene label appears within the ranked results.

### 4.2 Oxford5k mode

Oxford5k uses the standard relevance interpretation:

- `good ∪ ok` are treated as relevant;
- `junk` is ignored during evaluation.

For each query, the pipeline computes:

- AP (average precision)
- top-1 / top-5 / top-10 success
- MRR based on the first non-junk relevant result

The final summary reports **mAP** as the mean of per-query AP values.

The retrieval CSV also records dataset-aware metadata such as `query_name`, `query_image_id`, `match_image_id`, and, in Oxford mode, `is_relevant` / `is_junk` flags.

## 5. Results and Outputs

I ran the pipeline on the Oxford5k configuration (`dataset.mode: oxford5k`). The current run produced the following summary in `Assignment2/results/metrics.txt`:

- Queries: **55**
- Reference images: **5063**
- **mAP: 0.1032**
- **MRR: 0.6144**
- **Top-1 success: 0.6000**
- **Top-3 success: 0.6000**
- **Top-5 success: 0.6182**
- **Top-10 success: 0.6545**
- Average reference keypoints: **865.45**
- Average query keypoints: **800.22**

These results show that the current classical BoW pipeline can often place at least one relevant Oxford image near the top of the ranking, but the overall ranking quality is still limited, as reflected by the relatively low mAP. This is consistent with the known limitations of plain BoW retrieval on a challenging landmark benchmark: the representation captures local appearance reasonably well, but it is still sensitive to background clutter, repeated architectural patterns, and viewpoint variation.

A qualitative inspection of `Assignment2/results/retrieval_results.csv` also shows that some queries retrieve correct landmark images at very high ranks. For example, for `all_souls_1`, the first two ranked results are relevant (`all_souls_000013` and `all_souls_000015`), but several visually similar non-relevant Oxford building images also appear close to the top of the ranking. This pattern explains why top-1 and top-10 success are moderate while mAP remains lower.

Running the pipeline generates outputs in `Assignment2/results/`, including:

- `vocabulary.npy`
- `idf.npy`
- `database_histograms.npy`
- `retrieval_results.csv`
- `metrics.txt`
- `loop_closure.csv`
- ranking visualizations in `results/visualizations/`

In demo mode, the visualization titles show the query and top-ranked matches by scene. In Oxford mode, the visualizations show query names and annotate ranked results with `relevant` or `junk` when applicable.

## 6. Discussion

### Advantages

- The pipeline remains modular and easy to extend.
- Demo mode is preserved, so the original assignment flow still works.
- Oxford5k support makes the system usable on a real image retrieval benchmark.
- Query metadata is now explicit instead of inferred only from filenames.
- The implementation exports intermediate results and evaluation outputs for inspection.

### Limitations

- Hard assignment maps each descriptor to only one visual word, which can lose information near cluster boundaries.
- Pure BoW discards geometric layout, so false positives are still possible before spatial verification.
- Oxford5k is substantially larger than the demo set, so feature extraction and vocabulary construction are more computationally expensive.
- Loop closure detection is still based only on histogram similarity and does not use Oxford ground truth.

## 7. Possible Improvements

Several extensions would improve retrieval quality beyond the baseline:

- soft assignment of descriptors to multiple nearby words;
- spatial verification with stronger re-scoring after BoW ranking;
- inverted indexing for faster large-scale search;
- hierarchical vocabularies or vocabulary trees;
- VLAD or Fisher Vector aggregation for stronger global representations.

## 8. Conclusion

This assignment now implements a complete classical image retrieval pipeline with both a deterministic demo workflow and Oxford5k benchmark support. The code covers dataset preparation, query generation, BoW retrieval, spatial re-ranking, evaluation, and output export. Although the method is classical compared with modern deep retrieval systems, it remains an effective and interpretable baseline for image retrieval and loop closure detection.
