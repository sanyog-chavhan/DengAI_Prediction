# 🦟 Dengue Fever Prediction using Machine Learning

This repository presents a **machine learning approach** for predicting dengue fever outbreaks in **San Juan, Puerto Rico, and Iquitos, Peru**, leveraging climate and demographic data. The project applies **regression models** to estimate total dengue cases per city, year, and week, helping **public health officials and policymakers** proactively respond to dengue outbreaks.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)  
2. [Repository Structure](#-repository-structure)  
3. [Data Description](#-data-description)  
4. [Methodology](#-methodology)  
5. [Usage Instructions](#-usage-instructions)  
6. [Requirements](#-requirements)  
7. [Results](#-results)  
8. [Contributing](#-contributing)  

---

## 🔍 Project Overview

- **Goal**: Predict total dengue cases based on historical climate and demographic data.  
- **Dataset**: Contains climate factors, population demographics, and past dengue occurrences.  
- **Models Used**: Decision Trees, Random Forest, Gradient Boosting, XGBoost.  
- **Hyperparameter Tuning**: GridSearchCV, Optuna, TPOT AutoML.  

### 🎯 Key Objectives
1. **Regression**: Forecast total dengue cases per city and time frame.  
2. **Feature Engineering**: Handle missing values, outliers, and categorical encoding.  
3. **Evaluation Metrics**:  
   - **Mean Squared Error (MSE)**  
   - **Mean Absolute Error (MAE)**  
   - **R-squared (R²)**  

---

## 📂 Repository Structure
```
DengAI_Prediction/
├── notebooks/
│   ├── Dengue_Cases_Prediction.ipynb  # Main Jupyter Notebook for training models
├── Dengue_Cases_Prediction.pdf        # Research paper explaining methodology
├── requirements.txt                    # Dependencies required for the project
├── README.md                           # Project documentation
```

---

## 📊 Data Description

- **Dataset Size**: 1456 instances (train) + 416 instances (test).  
- **Feature Categories**:  
  - **Climate Data**: Temperature, humidity, precipitation.  
  - **Demographic Data**: Population density, healthcare accessibility.  
- **Target Variable**: `total_cases` (continuous variable).  

---

## ⚙️ Methodology

### **Data Preprocessing & Feature Engineering**
- **Missing Values** → Median imputation.  
- **Categorical Encoding** → City labels converted to numerical values.  
- **Feature Scaling** → Standardization using `StandardScaler`.  
- **Outlier Removal** → Handled using interquartile range (IQR).  

### **Regression Models**
- **Decision Tree Regressor** → Baseline model.  
- **Random Forest Regressor** → Reduces variance through ensemble learning.  
- **Gradient Boosting Regressor** → Sequential learning to reduce errors.  
- **XGBoost Regressor** → Optimized with column subsampling and regularization.  

### **Hyperparameter Tuning**
- **GridSearchCV** → Exhaustive search over hyperparameters.  
- **Optuna** → Bayesian optimization for faster tuning.  
- **TPOT AutoML** → Genetic algorithms for pipeline optimization.  

---

## 🚀 Usage Instructions

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/sanyog-chavhan/DengAI_Prediction.git
cd DengAI_Prediction
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Jupyter Notebook**
```bash
jupyter notebook
```
- Open `notebooks/Dengue_Cases_Prediction.ipynb`.  

### **4️⃣ Explore Results**
- The notebook contains visualizations & evaluation metrics for each model.  
- Refer to the `Dengue_Cases_Prediction.pdf` for a detailed breakdown.  

---

## 📦 Requirements

Below is a typical `requirements.txt`:

```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
optuna
tpot
jupyter
```

---

## 📈 Results

### **Regression Model Performance**

| Model            | MSE   | MAE   | R² Score |
|-----------------|------|------|---------|
| **Decision Tree**  | 0.597 | 0.557 | 0.629   |
| **Random Forest**  | 0.363 | 0.472 | 0.774   |
| **Gradient Boosting**  | 0.454 | 0.521 | 0.717   |
| **XGBoost**  | 0.364 | 0.462 | 0.774   |
| **Gradient Boosting (Optuna)**  | **0.297** | **0.426** | **0.815**  |

### **Best Model**: Gradient Boosting (Optimized via Optuna)  
Achieved the highest R² score of **0.815**, making it the most effective predictor of dengue outbreaks.

---

## 🤝 Contributing

1. **Fork** this repo.  
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new model or fix bug"
   ```
4. Push to GitHub:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a **Pull Request**.
