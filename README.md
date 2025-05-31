# README

## Project Overview

This project consists of three main scripts for structuring, analyzing, and modeling risk-related data collected from two tools: **Geslin** and **Rutisafenet**. The goal is to predict risk behaviors using different machine learning models.

---

## 1. `Final_structuring_dataset`

This script is responsible for **structuring the dataset**. It works **exclusively** with the original data files **directly downloaded from the Geslin and Rutisafenet tools**. The dataset must have a specific structure, and only these exact files should be used as input:

- **Geslin**:  
  - Provides *risk categories* (risk behaviors) and *sociodemographic data*.
- **Rutisafenet**:  
  - Provides *risk factors*.

The script merges and structures these datasets into a unified, clean format ready for analysis and modeling.

---

## 2. `Studying_the_dataset`

This script performs an **exploratory data analysis** on the structured datasets created in the previous step. Key functions include:

- Data distribution analysis
- Correlation analysis
- Analysis of missing values
- Other exploratory visualizations

Its purpose is to understand the data and its main features before building predictive models.

---

## 3. `Prediction_experiments`

This script runs **prediction experiments** using the structured datasets. The main objectives are:

- **Predicting risk behaviors (from Geslin):**
  - **Risk / No risk**
  - **Type of risk**: Aggressive, Self-harm, Absconding, or No risk
- **Input features include:**
  - Risk factors (from Rutisafenet)
  - Sociodemographic indicators (from Geslin)

The following machine learning models are applied:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Simple Neural Network**

Experiments are performed only on the datasets structured in Step 1.

---

## Requirements

- You must use the **original data files** exported from **Geslin** and **Rutisafenet** (not any other or modified sources).
- Python packages required: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow/keras` (or similar).

---

## Workflow

1. **Run** `Final_structuring_dataset` to build the structured datasets from raw data.
2. **Run** `Studying_the_dataset` to explore and analyze the structured data.
3. **Run** `Prediction_experiments` to train and evaluate models on the structured data.

---

*Let me know if you need more details, example usage, or a section on installation!*
