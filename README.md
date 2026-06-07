# California Housing Price Prediction

A comprehensive machine learning project that predicts California housing prices using statistical learning methods and regression models implemented primarily from scratch with NumPy.

This project explores the complete machine learning workflow, including data preprocessing, feature engineering, model development, hyperparameter tuning, evaluation, and visualization. It compares Ordinary Least Squares (OLS), Ridge Regression, and Polynomial Ridge Regression to understand the impact of regularization and feature expansion on predictive performance.

---

## Overview

Housing price prediction is a classic regression problem that combines statistical analysis, machine learning, and feature engineering. This project uses the California Housing dataset to build and evaluate predictive models while demonstrating key concepts from mathematical methods in data analysis and machine learning.

The implementation focuses on understanding the mathematical foundations behind regression models rather than relying entirely on machine learning libraries.

---

## Features

* Exploratory Data Analysis (EDA)
* Data Cleaning and Preprocessing
* Feature Engineering
* Train / Validation / Test Splitting
* Feature Standardization
* Ordinary Least Squares Regression (MLE)
* Ridge Regression (MAP)
* Cross-Validation for Hyperparameter Selection
* Polynomial Feature Expansion (Degree 2)
* Model Evaluation and Comparison
* Publication-Quality Visualizations

---

## Dataset

The project uses the California Housing dataset containing demographic, geographic, and housing-related information from California census districts.

### Input Features

* Longitude
* Latitude
* Housing Median Age
* Total Rooms
* Total Bedrooms
* Population
* Households
* Median Income
* Ocean Proximity

### Target Variable

* Median House Value

To reduce skewness and improve model performance, the target variable is transformed using:

```math
log(1 + median_house_value)
```

---

## Data Preprocessing

The preprocessing pipeline includes:

### Missing Value Handling

Rows containing missing values in the `total_bedrooms` feature are removed.

### Feature Engineering

Three additional features are created:

* Rooms per Household
* Bedrooms per Room
* Population per Household

### Categorical Encoding

The `ocean_proximity` feature is transformed using one-hot encoding.

### Feature Standardization

All numerical features are standardized using training-set statistics only to prevent data leakage.

---

## Models Implemented

### 1. Baseline Predictor

A simple mean-value predictor used as a reference model.

### 2. Ordinary Least Squares (OLS)

Implemented using the normal equation:

```math
\theta_{MLE} = (X^T X)^{-1} X^T y
```

This serves as the primary linear regression benchmark.

### 3. Ridge Regression

Ridge Regression introduces L2 regularization:

```math
\theta_{MAP} = (X^T X + \lambda I)^{-1} X^T y
```

Benefits include:

* Reduced overfitting
* Improved generalization
* More stable coefficient estimates

### 4. Polynomial Ridge Regression

Degree-2 polynomial features are generated to capture nonlinear relationships between variables.

The expanded feature space includes:

* Original features
* Squared terms
* Pairwise interaction terms

The resulting model is then trained using Ridge Regression.

---

## Cross-Validation

The regularization parameter λ is selected using:

* 5-Fold Cross Validation
* Validation RMSE minimization

Search range:

```text
λ ∈ [10⁻⁴, 10³]
```

This ensures that the chosen model balances bias and variance effectively.

---

## Evaluation Metrics

Model performance is measured using:

### Root Mean Squared Error (RMSE)

Measures the average magnitude of prediction errors.

### Mean Absolute Error (MAE)

Measures the average absolute prediction error.

### R² Score

Measures how much variance in the target variable is explained by the model.

---

## Visualizations Generated

The project automatically generates and saves the following figures:

### Figure 1 — Exploratory Data Analysis

* Price distribution
* Log-price distribution
* Income vs. house value relationship
* Correlation heatmap

### Figure 2 — Cross-Validation Results

* RMSE versus λ

### Figure 3 — Regularization Path

* Coefficient shrinkage as λ increases

### Figure 4 — Predicted vs Actual Values

* OLS
* Ridge
* Polynomial Ridge

### Figure 5 — Residual Analysis

* Residual distributions and patterns

### Figure 6 — Learning Curve

* Training vs validation RMSE

### Figure 7 — Model Comparison

* RMSE
* MAE
* R²

### Figure 8 — Feature Importance

* Largest Ridge Regression coefficients

---

## Project Structure

```text
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
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/california-housing-price-prediction.git

cd california-housing-price-prediction
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn
```

---

## Usage

Place the dataset inside the `data/` directory:

```text
data/housing.csv
```

Run the analysis:

```bash
python analysis.py
```

All visualizations will be saved automatically in the `figures/` directory.

---

## Learning Outcomes

This project demonstrates practical applications of:

* Linear Algebra
* Statistical Learning
* Regression Analysis
* Maximum Likelihood Estimation (MLE)
* Maximum A Posteriori Estimation (MAP)
* Regularization Techniques
* Cross Validation
* Hyperparameter Tuning
* Feature Engineering
* Bias–Variance Tradeoff
* Model Evaluation

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

## Future Improvements

Potential extensions include:

* Lasso Regression
* Elastic Net Regression
* Gradient Descent Optimization
* Random Forest Regression
* Gradient Boosting Methods
* XGBoost
* Neural Networks
* Geographic Feature Enrichment
* Automated Hyperparameter Optimization

---

## Author

**Raza Ur Rehman**

Bachelor's Student in Molecular Biology and Genetics
Abdullah Gul University

---

## License

This project is licensed under the MIT License and is available for educational, academic, and research purposes.
