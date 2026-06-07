California Housing Price Prediction

A comprehensive machine learning project developed for Mathematical Methods in Data Analysis and Machine Learning at Abdullah Gul University.

This project predicts California housing prices using statistical learning techniques and compares the performance of:

Ordinary Least Squares (OLS) Regression (Maximum Likelihood Estimation)
Ridge Regression (Maximum A Posteriori Estimation)
Polynomial Feature Expansion (Degree 2) with Ridge Regression

Unlike many machine learning projects that rely heavily on libraries such as Scikit-Learn, the core regression models in this implementation are built directly using mathematical formulations and linear algebra operations from NumPy.

Project Objectives

The main goals of this project are:

Perform exploratory data analysis on housing market data
Clean and preprocess real-world datasets
Engineer meaningful predictive features
Implement regression models from scratch
Apply regularization techniques to reduce overfitting
Use cross-validation for hyperparameter tuning
Compare model performance using multiple evaluation metrics
Visualize model behavior and prediction quality
Dataset

The project uses the California Housing Dataset, which contains demographic, geographic, and housing-related information collected from California census districts.

Features Include
Median Income
Housing Median Age
Total Rooms
Total Bedrooms
Population
Households
Latitude
Longitude
Ocean Proximity
Target Variable
Median House Value

The target variable is log-transformed to reduce skewness and improve model performance.

Methodology
1. Data Exploration

The project performs:

Distribution analysis of housing prices
Log-price distribution visualization
Feature correlation analysis
Income vs. housing value investigation
2. Data Cleaning

The preprocessing pipeline includes:

Handling missing values
One-hot encoding categorical variables
Creating engineered features
Log transformation of housing prices
Engineered Features
Rooms per Household
Bedrooms per Room
Population per Household
3. Train / Validation / Test Split

Dataset is divided into:

Dataset	Percentage
Training	70%
Validation	15%
Testing	15%

Random shuffling is performed using NumPy for reproducibility.

4. Feature Standardization

Features are standardized using training-set statistics only to prevent data leakage.

5. Ordinary Least Squares (MLE)

Implemented using the Normal Equation:

θ
MLE
	​

=(X
T
X)
−1
X
T
y

This serves as the baseline linear regression model.

6. Ridge Regression (MAP)

Implemented using L2 regularization:

θ
MAP
	​

=(X
T
X+λI)
−1
X
T
y

Benefits include:

Reduced overfitting
Improved generalization
More stable coefficient estimates
7. Cross-Validation

The optimal regularization parameter λ is selected using:

5-Fold Cross Validation
Validation RMSE minimization

Search space:

λ ∈ [10⁻⁴, 10³]
8. Polynomial Feature Expansion

The project expands selected features to degree-2 polynomial space:

Original features
Squared terms
Pairwise interaction terms

This allows the model to capture nonlinear relationships in housing prices.

Evaluation Metrics

Model performance is measured using:

Root Mean Squared Error (RMSE)

RMSE=
n
1
	​

∑
i=1
n
	​

(y
i
	​

−
y
^
	​

i
	​

)
2
	​


Mean Absolute Error (MAE)

Measures average prediction error magnitude.

Coefficient of Determination (R²)

Measures the proportion of variance explained by the model.

Generated Visualizations

The project automatically creates and saves:

Figure 1 — Exploratory Data Analysis
Housing price distribution
Log-price distribution
Income vs price scatter plot
Correlation heatmap
Figure 2 — Cross Validation Results
RMSE vs λ
Figure 3 — Ridge Regularization Path
Coefficient shrinkage across λ values
Figure 4 — Predicted vs Actual Values
OLS
Ridge
Polynomial Ridge
Figure 5 — Residual Analysis
Residual plots for all models
Figure 6 — Learning Curve
Training vs Validation RMSE
Figure 7 — Model Comparison
RMSE
MAE
R² comparison
Figure 8 — Feature Importance
Largest Ridge Regression coefficients
Project Structure
project/
│
├── data/
│   └── housing.csv
│
├── figures/
│   ├── fig1_eda.png
│   ├── fig2_cv_lambda.png
│   ├── fig3_regularisation_path.png
│   ├── fig4_predicted_vs_actual.png
│   ├── fig5_residuals.png
│   ├── fig6_learning_curve.png
│   ├── fig7_model_comparison.png
│   └── fig8_coefficients.png
│
├── analysis.py
│
└── README.md
Installation

Clone the repository:

git clone https://github.com/yourusername/california-housing-price-prediction.git

cd california-housing-price-prediction

Install required packages:

pip install numpy pandas matplotlib seaborn
Usage

Place the California Housing dataset inside the data/ directory:

data/housing.csv

Run the analysis:

python analysis.py

All figures and evaluation results will be generated automatically.

Key Learning Concepts

This project demonstrates practical applications of:

Linear Regression
Maximum Likelihood Estimation (MLE)
Maximum A Posteriori Estimation (MAP)
Regularization
Feature Engineering
Cross Validation
Model Selection
Bias-Variance Tradeoff
Polynomial Regression
Statistical Learning Theory
Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Results

The project compares:

Baseline Mean Predictor
OLS Regression
Ridge Regression
Polynomial Ridge Regression

Performance is evaluated using held-out test data to determine which approach generalizes best for housing price prediction.

Future Improvements

Potential extensions include:

Lasso Regression
Elastic Net Regularization
Gradient Descent Optimization
Random Forest Regressors
Gradient Boosting Methods
Neural Networks
Geographic Feature Enrichment
Automated Hyperparameter Optimization
Author

Raza Ur Rehman
Bachelor's in Molecular Biology and Genetics
Abdullah Gul University

License

This project is released under the MIT License. Feel free to use, modify, and extend it for educational and research purposes.
