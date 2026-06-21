# House Price Prediction

A machine learning project that predicts house sale prices using Linear Regression, Ridge Regression, and Lasso Regression, with a complete data preprocessing pipeline applied to a real-world Kaggle dataset.

## Overview

This project walks through the full machine learning workflow from raw, messy data to a trained, evaluated model, using the Kaggle House Prices: Advanced Regression Techniques dataset.

## Dataset

- Rows: 1,460 houses
- Features: 80 (mix of numerical and categorical)
- Target: SalePrice

## Workflow

1. Missing Value Handling
   - Distinguished between "feature doesn't exist" (e.g., no pool, no alley) and genuinely missing data
   - Filled structural absences with "None", and true missing values with median (numerical) or mode (categorical)

2. Target Transformation
   - SalePrice was right-skewed (skewness about 1.88)
   - Applied log1p transformation to normalize the distribution (skewness reduced to about 0.12)

3. Feature Encoding
   - Ordinal encoding for quality-scale features with natural order (e.g., ExterQual, KitchenQual)
   - One-Hot encoding for nominal categorical features with no inherent order (e.g., Neighborhood, MSZoning)
   - Final feature set: 232 columns

4. Feature Scaling
   - Applied StandardScaler, fit only on training data to prevent data leakage

5. Modeling and Evaluation
   - Trained and compared three models using 5-fold Cross-Validation:

   Model: Linear Regression, Test R2: 0.772, CV Mean R2: 0.736
   Model: Ridge Regression, Test R2: 0.791, CV Mean R2: 0.754
   Model: Lasso Regression, Test R2: 0.844, CV Mean R2: 0.802

6. Feature Importance Analysis
   - Lasso reduced 229 features to 170 by zeroing out irrelevant ones
   - Cross-checked top features against correlation analysis to avoid misleading conclusions from rare, low-frequency categories

## Key Takeaways

- Lasso outperformed Linear and Ridge regression due to its built-in feature selection, which reduced overfitting from noisy one-hot encoded features
- OverallQual and GrLivArea were consistently the strongest predictors across both correlation analysis and model coefficients
- Demonstrated the importance of cross-checking feature importance results rather than trusting a single metric

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (LinearRegression, Ridge, Lasso, StandardScaler, train_test_split, cross_val_score)
- Matplotlib, Seaborn

## Files

- 02_House_price_prediction.ipynb: full analysis and modeling notebook

## How to Run

1. Clone this repository
2. Open 02_House_price_prediction.ipynb in Jupyter Notebook or Google Colab
3. Download the dataset from Kaggle and place train.csv in the same directory
4. Run all cells
