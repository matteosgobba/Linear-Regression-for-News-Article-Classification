# 📰 News Article Classification

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains the final project for the **Data Science and Machine Learning Lab** course (Master's Degree in Data Science and Engineering) at **Politecnico di Torino**. 

The goal of this project is to build a robust machine learning classification pipeline capable of automatically categorizing online news articles into 7 predefined thematic categories based on their textual content and metadata.

## 📑 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
  - [Text Representation](#text-representation)
  - [Modeling](#modeling)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Author](#-author)

## 🔍 Overview
With the proliferation of digital news, automated categorization is critical for content recommendation and efficient news aggregation. This project tackles the multi-class classification of ~100,000 international news articles using advanced TF-IDF representations and linear classifiers. 

The models were rigorously tuned via Grid Search cross-validation, specifically optimizing for the **Macro-F1 score** to account for the inherent class imbalance in the dataset. The final model (**LinearSVC**) achieved a top-25 placement on the competitive course leaderboard.

## 📊 Dataset
The provided dataset consists of approximately 100,000 records partitioned into:
* `development.csv` (~80,000 instances) - Used for training and validation.
* `evaluation.csv` (~20,000 instances) - Unlabeled set used for leaderboard evaluation.

**Target Categories:**
0. International News
1. Business
2. Technology
3. Entertainment
4. Sports
5. General News
6. Health

**Features:**
* `id`: Unique identifier
* `source`: Publisher/News outlet
* `title`: Article title
* `article`: Full textual content
* `page_rank`: Source page rank
* `timestamp`: Date and time of publication

## 🧠 Methodology

### Preprocessing & Feature Engineering
* **Data Cleaning:** Unescaped HTML entities and filtered out placeholder noise (e.g., `\N` values).
* **Feature Creation:** Extracted word counts for titles (`title_wc`) and articles (`article_wc`).
* **Temporal Features:** Handled missing/placeholder timestamps by creating a boolean indicator (`ts_missing`) and parsed valid timestamps into `year`, `month`, `dayofweek`, and `hour`.
* **Text Unification:** Concatenated `source`, `title`, and `article` into a single, rich `text` field to capture both publisher-specific styles and semantic content.

### Text Representation
Given the high dimensionality and sparsity of the data, a Bag-of-Words approach was adopted:
* **Word-level TF-IDF:** Extracted unigrams and bigrams `(1, 2)`.
* **Character-level TF-IDF:** Extracted character n-grams `(3, 5)` (Used primarily for LinearSVC to capture morphological patterns).
* Applied sublinear term frequency scaling (`1 + log(tf)`) and L2 normalization. Stop words were removed, and terms appearing in less than 2 documents were ignored (`min_df=2`).

### Modeling
Three main linear classifiers were evaluated, all leveraging the `class_weight='balanced'` parameter to mitigate the effects of the underrepresented "Health" category:
1. **Linear Support Vector Classifier (LinearSVC):** *[Best Model]* Maximizes the margin between classes; highly effective in high-dimensional, sparse text spaces.
2. **Logistic Regression:** Probabilistic linear model providing strong, well-calibrated performance.
3. **Stochastic Gradient Descent (SGDClassifier):** Scalable baseline optimizing a hinge loss function.

Models were optimized using a multi-step `GridSearchCV` with Stratified 3-Fold Cross-Validation, ensuring strict reproducibility (`random_state=42`).

## 🏆 Results

Model performances were evaluated using the **Macro-F1 score**. The table below summarizes the Cross-Validation scores on the development set and the final unseen test scores on the public leaderboard.

| Model | Best Hyperparameters | CV Macro-F1 | Leaderboard Macro-F1 |
| :--- | :--- | :---: | :---: |
| **LinearSVC** (Final) | `C=0.1` | **0.7109** | **0.735** |
| **Logistic Regression** | `C=2`, `solver='saga'` | 0.7053 | 0.728 |
| **SGDClassifier** | `alpha=1e-5`, `loss='hinge'` | 0.6888 | 0.698 |
| *Dummy (Random)* | `strategy='stratified'` | 0.1417 | 0.150 |

*Note: The final LinearSVC solution ranked in the Top 25 of the course leaderboard, maintaining strong generalization with no signs of overfitting.*

## 📂 Repository Structure

```text
├── data/
│   ├── development.csv         # Training and validation dataset
│   └── evaluation.csv          # Unlabeled dataset for final predictions
├── project_notebook_sgobba_matteo.ipynb # Main Jupyter Notebook with full pipeline code
├── Report_Matteo_Sgobba.pdf    # Detailed IEEE-format technical report
├── submissions/                # Generated CSV prediction files
│   ├── submission1.csv         # LinearSVC predictions
│   └── submission2.csv         # Logistic Regression predictions
└── README.md                   # Project documentation
```

## 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/yourusername/news-article-classification.git
cd news-article-classification
```

2. Install Dependencies

Ensure you have Python 3.8+ installed. Install the required libraries using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

3. Execute the Notebook

Launch Jupyter Notebook or Jupyter Lab to interact with the code:

```bash
jupyter notebook project_notebook_sgobba_matteo.ipynb
```

Note: The notebook caches preprocessing steps using `joblib.Memory` to significantly speed up hyperparameter tuning runs.

## Author

Matteo Sgobba

- M.Sc. Data Science and Engineering @ Politecnico di Torino
- Contact: matteo.sgobba@studenti.polito.it
- LinkedIn: https://www.linkedin.com/in/matteosgobba/

