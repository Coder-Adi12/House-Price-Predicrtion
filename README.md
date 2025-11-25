# 🏠 House Price Prediction - Advanced Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Top%2040%25-blue)

## 📌 Project Overview
This project predicts residential home prices in Ames, Iowa. The goal was to build a machine learning model capable of handling complex, real-world data with 79 explanatory variables (features) describing almost every aspect of residential homes.

Using advanced regression techniques (**Lasso, Ridge, and ElasticNet**), I achieved a **Kaggle Leaderboard Score (RMSE) of 0.141**, placing in the top percentile of submissions.

## 📂 Dataset
The dataset comes from the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition.
**Train.csv:** 1460 rows, 81 columns (Features + Target)
* **Test.csv:** 1459 rows, 80 columns (Features only)

## 🛠️ Key Technical Steps

### 1. Data Cleaning & Preprocessing
* **Handling Missing Values:**
    * Categorical features (e.g., `PoolQC`, `Alley`) were filled with "None" to indicate the absence of the feature.
    * Numerical features (e.g., `GarageArea`) were filled with `0` where appropriate.
    * Statistical imputation (Median/Mode) was used for `LotFrontage` and `Electrical`.
* **Feature Engineering:**
    * **Manual Ordinal Mapping:** Preserved the rank of ordinal features (e.g., `ExterQual`, `KitchenQual`) by mapping them to an integer scale (0-5) instead of One-Hot Encoding.
    * **One-Hot Encoding:** Applied to nominal variables (e.g., `Neighborhood`, `RoofStyle`).
* **Target Transformation:**
    * Applied `np.log1p()` (Log Transformation) to the `SalePrice` to correct skewness and normalize the data distribution.

### 2. Modeling & Regularization
I implemented and compared multiple linear algorithms to handle the high dimensionality (200+ features after encoding) and prevent overfitting.

* **Linear Regression:** Served as a baseline (suffered from high variance/overfitting).
* **Ridge Regression (L2):** Handled multicollinearity by shrinking coefficients.
* **Lasso Regression (L1):** Performed feature selection by shrinking less important feature coefficients to zero.
* **ElasticNet:** Combined L1 and L2 penalties.

### 3. Evaluation
I used **5-Fold Cross-Validation** with **RMSE (Root Mean Squared Error)** as the scoring metric.

| Model | CV RMSE Score |
| :--- | :--- |
| Ridge Regression | 0.1360 |
| **Lasso Regression** | **0.1338 (Best)** |
| ElasticNet | 0.1339 |

**Final Submission:** The Lasso model was retrained on the full dataset and achieved a **0.14139** score on the Kaggle Test set.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/House-Price-Prediction.git](https://github.com/YourUsername/House-Price-Prediction.git)