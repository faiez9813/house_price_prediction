\## House Price Prediction



A machine learning project that predicts house sale prices using Linear Regression, Ridge Regression, and Lasso Regression, with a complete data preprocessing pipeline applied to a real-world Kaggle dataset.



\## Overview



This project walks through the full machine learning workflow — from raw, messy data to a trained, evaluated model — using the \[Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset.



\## Dataset



\- \*\*Rows:\*\* 1,460 houses

\- \*\*Features:\*\* 80 (mix of numerical and categorical)

\- \*\*Target:\*\* `SalePrice`



\## Workflow



1\. \*\*Missing Value Handling\*\*

&#x20;  - Distinguished between "feature doesn't exist" (e.g., no pool, no alley) and genuinely missing data

&#x20;  - Filled structural absences with `"None"`, and true missing values with median (numerical) or mode (categorical)



2\. \*\*Target Transformation\*\*

&#x20;  - `SalePrice` was right-skewed (skewness ≈ 1.88)

&#x20;  - Applied `log1p` transformation to normalize the distribution (skewness reduced to ≈ 0.12)



3\. \*\*Feature Encoding\*\*

&#x20;  - \*\*Ordinal encoding\*\* for quality-scale features with natural order (e.g., `ExterQual`, `KitchenQual`)

&#x20;  - \*\*One-Hot encoding\*\* for nominal categorical features with no inherent order (e.g., `Neighborhood`, `MSZoning`)

&#x20;  - Final feature set: 232 columns



4\. \*\*Feature Scaling\*\*

&#x20;  - Applied `StandardScaler`, fit only on training data to prevent data leakage



5\. \*\*Modeling \& Evaluation\*\*

&#x20;  - Trained and compared three models using 5-fold Cross-Validation:



&#x20;  | Model | Test R² | CV Mean R² |

&#x20;  |---|---|---|

&#x20;  | Linear Regression | 0.772 | 0.736 |

&#x20;  | Ridge Regression | 0.791 | 0.754 |

&#x20;  | \*\*Lasso Regression\*\* | \*\*0.844\*\* | \*\*0.802\*\* |



6\. \*\*Feature Importance Analysis\*\*

&#x20;  - Lasso reduced 229 features to 170 by zeroing out irrelevant ones

&#x20;  - Cross-checked top features against correlation analysis to avoid misleading conclusions from rare, low-frequency categories



\## Key Takeaways



\- Lasso outperformed Linear and Ridge regression due to its built-in feature selection, which reduced overfitting from noisy/irrelevant one-hot encoded features

\- `OverallQual` and `GrLivArea` were consistently the strongest predictors across both correlation analysis and model coefficients

\- Demonstrated the importance of cross-validating feature importance results rather than trusting a single metric



\## Tech Stack



\- Python, Pandas, NumPy

\- Scikit-learn (LinearRegression, Ridge, Lasso, StandardScaler, train\_test\_split, cross\_val\_score)

\- Matplotlib, Seaborn



\## Files



\- `02\_House\_price\_prediction.ipynb` — full analysis and modeling notebook



\## How to Run



1\. Clone this repository

2\. Open `02\_House\_price\_prediction.ipynb` in Jupyter Notebook or Google Colab

3\. Download the dataset from \[Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and place `train.csv` in the same directory

4\. Run all cells

