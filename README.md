# Customer Segmentation with K-Means Clustering

K-Means clustering applied to a transnational e-commerce dataset to identify
customer segments supporting marketing decision-making.

**Author:** Rudina
**Context:** University of Cambridge Data Science with ML & AI Career Accelerator — Mini-project, June 2026
**Course:** C101, Week 6

---

## Overview

An e-commerce company covering 47 countries across five continents needs
actionable customer segments to support targeted marketing and prioritise
high-value or at-risk customer groups. This project applies unsupervised
machine learning to segment 68,300 unique customers using five RFM-style
features engineered from transaction-level data.

## Dataset

- Source: SAS `CUSTOMERS_CLEAN` dataset (SAS, 2024)
- Size: 951,668 rows × 20 columns (transaction-level)
- After deduplication: 951,648 rows, representing 68,300 unique customers
- Time range: 1 January 2012 – 30 December 2016

## Methods

| Stage | Technique |
|---|---|
| Data cleaning | Currency and date parsing, duplicate removal |
| Feature engineering | 5 customer-level features: Frequency, Recency, Customer Lifetime Value (CLV), Average Unit Cost, Customer Age |
| Outlier detection | Interquartile Range (IQR) — outliers retained as genuine high-value customers |
| Preprocessing | Log transformation of skewed features (`np.log1p`) and standardisation via `ColumnTransformer` + `Pipeline` |
| Cluster count evaluation | Elbow method (WCSS), silhouette coefficient, Davies-Bouldin Index, Ward-linkage hierarchical clustering (dendrogram) |
| Final model | K-Means, k = 5, `random_state=42`, `n_init=10` |
| Dimensionality reduction | PCA and t-SNE (perplexity 5–30) |

## Key Results

- **k = 5 selected** for the final K-Means model, balancing statistical evidence (DBI reached its lowest value 1.289 at k=5) with business interpretability
- **PCA:** first two components explained 48.3% + 20.1% of total variance
- **t-SNE** provided better visual cluster separation than PCA in the 2D projection
- Clusters overlap in 2D space, suggesting higher-dimensional structure driven partly by the strong Frequency–CLV correlation (r = 0.9)
- Five distinct customer segments identified by size (17,946 / 16,780 / 16,619 / 8,758 / 8,197 customers)

## Repository Contents

- `RUDINA_CAM_C101_Week_6_Mini_project.ipynb` — full analysis notebook
- `RUDINA_CAM_C101_Week_6_Mini-project.pdf` — technical report

## Dependencies

- Python 3.10+
- pandas, numpy
- scikit-learn (KMeans, ColumnTransformer, StandardScaler, PCA, TSNE, silhouette_score, davies_bouldin_score)
- scipy (hierarchical clustering, dendrogram)
- matplotlib, seaborn

## Limitations & Future Work

- Overlapping clusters suggest higher-dimensional feature relationships not captured in 2D projections
- Strong collinearity between Frequency and CLV (r = 0.9) — feature selection or alternative dimensionality reduction could improve cluster separation
- Future iterations could test feature selection strategies and alternative clustering algorithms (DBSCAN, Gaussian Mixture Models)

## References

1. SAS, 2024. CUSTOMERS_CLEAN [Data set]. SAS. Last revised on 15 December 2021.
2. Słowiński et al., 2021. *Influence of Data Dimension Reduction, Feature Scaling and Activation Function on Machine Learning Performance.*
3. Xiao et al., 2023. *Optimizing graph layout by t-SNE perplexity estimation.*
