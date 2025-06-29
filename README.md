
# Liver Disease Prediction  AA1 Competition (UPC, 2025)

This repository contains the full project developed by **Matías Mora** and **Marc Cascant** for the AA1 Kaggle InClass competition at Universitat Politècnica de Catalunya (UPC), where we achieved **2nd place** on the public leaderboard.

##  Problem Overview

The goal was to predict whether a patient has liver disease using clinical and demographic data. The dataset, provided via Kaggle InClass, includes 579 samples with 10 features per subject. The binary classification task is complicated by a significant class imbalance: around 71% of samples correspond to liver disease patients.

##  Dataset

- **Source:** UCI Machine Learning Repository (Indian Liver Patient Dataset)
- **Features:** Age, Gender (binary), Total/Direct Bilirubin, Liver enzymes (SGPT, SGOT, AlkPhos), Proteins (TP, ALB), A/G ratio
- **Preprocessing:** The dataset was cleaned to remove missing values and encode categorical variables.

##  Project Structure

- `AA1_LiverDisease_Classification.ipynb`: Full code including EDA, preprocessing, model training, evaluation, and submission.
- `Informe_Final_Mora_Cascant.pdf`: Formal report describing methodology and results in IEEE format.
- `submission.csv`: Example of a valid Kaggle submission.

##  Methodology

We designed a **modular and extensible machine learning pipeline** using `scikit-learn` and `imbalanced-learn`, enabling flexible combination of preprocessing steps and models:

###  Exploratory Data Analysis (EDA)
- Analyzed feature distributions, outliers, and correlation structure.
- Found strong correlations (e.g. SGOT/SGPT, TB/DB).
- Handled skewness and non-normality via transformations (Yeo-Johnson).

###  Preprocessing
- **Correlation Handling:** Domain-informed feature engineering and ratio variables.
- **Scaling:** MinMax and RobustScaler used depending on model sensitivity.
- **Feature Selection:** SelectKBest and tree-based importance rankings.
- **Imbalance Handling:** SMOTE, Tomek Links, SMOTETomek, RandomOverSampler.

###  Models Tested
We evaluated a wide range of classifiers:
- **Linear models:** Logistic Regression, Linear Discriminant Analysis (LDA)
- **Probabilistic models:** QDA, Naive Bayes
- **Kernel models:** SVM (with and without custom kernels)
- **Instance-based:** KNN
- **Tree-based:** Decision Trees, Random Forest, Extra Trees
- **Boosting:** AdaBoost, Gradient Boosting, XGBoost, HistGradientBoosting
- **Bagging:** Custom ensembles and scikit-learn BaggingClassifier

###  Final Model: Custom Ensemble
After extensive experimentation (over 1000 pipeline configurations), the best results were achieved with a **soft-voting ensemble** that integrates:
- QDA (no resampling, MinMax scaling)
- LDA (Tomek Links, RobustScaler)
- KNN (SMOTETomek, RobustScaler)
- Logistic Regression (RandomOverSampler, RobustScaler)

Each classifier was embedded in its **own preprocessing pipeline**, avoiding data leakage and ensuring compatibility with cross-validation.

##  Evaluation

- **Cross-validated macro F1-score:** 0.71
- **Kaggle test F1-score:** 0.69
- **Recall for liver disease class:** ~0.73
- ROC AUC: 0.68

We prioritized **F1-score** due to class imbalance and emphasized **recall** for liver disease detection.

##  Lessons Learned

- Modular pipelines enable robust experimentation and reproducibility.
- Preprocessing is as important as model choice.
- Class imbalance requires careful treatment to avoid bias.
- Ensemble methods combining diverse classifiers offer strong generalization.

##  Technologies Used

- Python 3.11
- scikit-learn
- imbalanced-learn
- numpy, pandas, matplotlib, seaborn
- Jupyter Notebook

##  How to Run

1. Clone the repository  
2. Open `AA1_LiverDisease_Classification.ipynb`  
3. Run all cells (requires Python and necessary packages)  
4. Generate `submission.csv` for Kaggle upload

##  License

This project is released for academic and educational purposes. All rights reserved by the authors.

---
Developed with  by Matías Mora & Marc Cascant
