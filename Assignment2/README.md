其实系统已经成功提取了这份 PDF 中的文字内容！由于我已经可以读取这些文本，我为你整理了这份作业要求的详细文字描述和大纲。

这是一份关于**计算机视觉（Computer Vision）**的编程作业。以下是具体的内容提要：

### 📌 作业概述
* [cite_start]**作业名称**：Visual Bag of Words for Image Retrieval（基于视觉词袋模型的图像检索）[cite: 1]。
* [cite_start]**核心目标**：实现一个视觉词袋（BoW）模型，用于计算机视觉应用中的图像检索和/或闭环检测（Loop Closure Detection）[cite: 3][cite_start]。需要提取视觉特征、构建视觉词汇表，并将其用于匹配图像 [cite: 4]。
* [cite_start]**截止日期**：作业发布之日后的三周 [cite: 5]。

### 🎯 任务说明
[cite_start]你将获得一个包含 N 张参考图像的数据库，以及一张或多张用于查询的图像 [cite: 8, 9]。你需要实现一个流水线（Pipeline）来完成：
1.  [cite_start]**图像检索（Image Retrieval）**：为查询图像在数据库中寻找最相似的图像 [cite: 11]。
2.  [cite_start]**闭环检测（Loop Closure Detection）**：针对连续的图像输入（例如来自机器人或摄像头的视频流），识别摄像头何时回到了之前访问过的位置 [cite: 12]。

### ⚙️ 算法实现步骤 (Pipeline)
作业要求实现以下五个主要步骤：
1.  [cite_start]**特征提取（Feature Extraction）**：从所有图像中提取局部特征（如 SIFT 或 ORB）[cite: 15][cite_start]。你需要使用 OpenCV 实现 `extract_features` 函数，返回关键点（Keypoints）和一个 $N\times D$ 维度的特征描述子矩阵 [cite: 36, 38, 39, 47]。
2.  [cite_start]**词汇表创建（Vocabulary Creation）**：收集所有数据库图像的描述子，并使用 K-means 聚类算法来创建“视觉词汇” [cite: 51, 52, 53][cite_start]。这部分由 `build_vocabulary` 函数实现，返回聚类中心 [cite: 50, 63]。
3.  [cite_start]**图像表示（Image Representation）**：通过寻找最近邻的视觉词汇，将每张图像的描述子转换为词频直方图（建议应用 TF-IDF 权重）[cite: 66, 67, 68, 69][cite_start]。对应的函数是 `image_to_bow_histogram` [cite: 70]。
4.  [cite_start]**相似度匹配（Similarity Matching）**：实现 `compute_similarity` 函数，使用余弦相似度（Cosine）、L1、L2 或卡方距离（Chi-squared）来比较直方图 [cite: 83, 84, 86, 87, 88, 89]。
5.  [cite_start]**检索/检测（Retrieval/Detection）**：实现 `search_image` 和主流水线函数 `bow_retrieval_pipeline`，比较查询图像的直方图和数据库直方图，返回得分最高的前 `top_k` 张相似图像 [cite: 90, 92, 93, 95]。

### 🌟 进阶与加分项 (Bonus)
除了基本要求，作业还包含多个高级任务供挑战：
* [cite_start]**闭环检测**：实现 `detect_loop_closure`，通过维护一个近期图像的滑动窗口，并将当前图像与之前的图像进行比较（当相似度大于阈值时判定为闭环）[cite: 106, 108, 109, 110]。
* [cite_start]**其他进阶挑战**：包括软分配（Soft Assignment）、基于 RANSAC 的空间验证、倒排索引、词汇树（Vocabulary Tree）、VLAD/Fisher Vectors 聚合方法，以及数据库增量扩展 [cite: 139, 140, 141]。

### 📂 提交要求与评分标准
[cite_start]你需要提交一个包含以下内容的 ZIP 压缩包 [cite: 143]：
* [cite_start]**代码文件**：主实现代码 `bow_retrieval.py`、运行脚本 `run_retrieval.py`、依赖文件 `requirements.txt` 和配置文件 `config.yaml` [cite: 144, 145, 146, 147]。
* [cite_start]**数据和结果**：数据集文件（或下载说明），以及包含词汇表（`vocabulary.npy`）、检索结果（CSV）、性能指标（txt）和可视化图片的 `results/` 文件夹 [cite: 148, 149, 150, 151, 152, 153]。
* [cite_start]**报告文档**：一份不超过 3 页的 PDF 报告，讨论实现细节、参数选择、结果分析以及优缺点 [cite: 154, 155, 156, 157, 158]。

[cite_start]**评分标准**总计 100% [cite: 130]：
* [cite_start]特征提取 (20%) [cite: 132]
* [cite_start]词汇表创建 (20%) [cite: 133]
* [cite_start]图像表示 (20%) [cite: 134]
* [cite_start]检索准确率 (20%) [cite: 135]
* [cite_start]代码质量 (10%) [cite: 136]
* [cite_start]报告与分析 (10%) [cite: 137]