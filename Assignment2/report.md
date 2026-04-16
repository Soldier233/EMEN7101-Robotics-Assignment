# Visual Bag of Words for Image Retrieval

## 1. Objective

This assignment implements a classical Visual Bag of Words (BoW) pipeline for image retrieval. The system extracts local image descriptors, clusters them into a visual vocabulary, converts each image into a TF-IDF weighted histogram, and retrieves the most similar reference images for each query. The main experiment is conducted on the Oxford5k dataset.

The implementation is mainly organized in:
- `bow_retrieval.py`: feature extraction, vocabulary construction, BoW encoding, retrieval, evaluation, and result export;
- `run_retrieval.py`: dataset preparation, Oxford5k download/parsing, and experiment execution;
- `config.yaml`: configuration of feature extraction, vocabulary size, retrieval settings, and outputs.

## 2. Implementation Details

### 2.1 Feature extraction

Each image is converted to grayscale and processed with **SIFT**. SIFT is used because it is relatively robust to scale and rotation changes, which is important for landmark retrieval. The extracted local descriptors provide the basis for later visual word assignment.

### 2.2 Vocabulary construction and image representation

Descriptors from the reference images are pooled and clustered by **MiniBatchKMeans**. The cluster centers are treated as visual words. Each image is then represented as a histogram of visual word occurrences.

To improve discrimination, the histogram is weighted by **TF-IDF**. Frequent words that appear in many images are down-weighted, while more distinctive words contribute more strongly. The final BoW vector is L2-normalized before matching.

### 2.3 Retrieval and evaluation

For a query image, its BoW histogram is compared with all reference histograms. The main experiment uses **cosine similarity**, which is suitable for normalized TF-IDF vectors. The system ranks database images by similarity score and writes results to `retrieval_results.csv` and `performance_metrics.txt`.

The implementation also computes quantitative metrics such as mAP, MRR, Top-1 success, Top-5 success, and Top-10 success on Oxford5k.

## 3. Parameter Choices and Justifications

Several parameter choices are important for the current system. First, **SIFT** is selected instead of ORB because it is generally more stable for landmark retrieval under viewpoint and scale changes. Second, **MiniBatchKMeans** is used to build the vocabulary because it is more computationally practical than standard K-means for a large real dataset such as Oxford5k.

The retrieval stage uses **cosine similarity** because the BoW vectors are TF-IDF weighted and L2-normalized, making cosine comparison a natural choice. The pipeline also limits the displayed matches to the top-ranked results, which is consistent with retrieval evaluation and qualitative inspection. Overall, these settings were chosen to provide a reasonable balance between retrieval quality and computational cost.

## 4. Results Analysis

The latest Oxford5k run in `Assignment2/results/metrics.txt` reports:

- Queries: **55**
- Reference images: **5063**
- **mAP: 0.1032**
- **MRR: 0.6144**
- **Top-1 success: 0.6000**
- **Top-3 success: 0.6000**
- **Top-5 success: 0.6182**
- **Top-10 success: 0.6545**

These results show that the BoW system can often retrieve at least one relevant image near the top of the ranking, as indicated by the Top-1 and Top-10 success rates. The MRR is also reasonably high, which suggests that a relevant image is often found early in the ranked list.

The qualitative output in `Assignment2/results/performance_metrics.txt` is consistent with these summary metrics. For example, `all_souls_1.jpg` retrieves two correct matches in the top 3 with Precision@1 = 1.0 and Precision@5 = 0.4, while more difficult queries such as `ashmolean_1.jpg` and `balliol_5.jpg` fail to retrieve a relevant image at rank 1. This shows that the system works well for some queries but remains unstable across the full benchmark.

However, the overall ranking quality is still limited because the mAP is low. This indicates that although the system may return a correct match near the top, it does not rank all relevant images consistently well across the full list. Therefore, the method works as a baseline, but it is still not strong enough for highly accurate large-scale landmark retrieval.

## 5. Discussion of Limitations and Improvements

A main limitation of the classical BoW model is that it relies only on local visual word statistics and does not represent spatial layout explicitly. On Oxford5k, many buildings contain repeated windows, edges, and textures, so visually similar but irrelevant images can receive high similarity scores.

Another limitation is quantization error from the visual vocabulary. Different local features may be mapped to the same visual word, which reduces discriminative power. In addition, the system depends strongly on parameter settings such as vocabulary size and the number of extracted features.

Possible improvements include adding stronger **spatial verification**, tuning the vocabulary size more carefully, or replacing the classical BoW representation with more advanced learned image features. These changes could improve ranking precision and reduce false matches on difficult landmark queries.

## 6. Declaration

All reported results are generated by my own implementation in this assignment.
