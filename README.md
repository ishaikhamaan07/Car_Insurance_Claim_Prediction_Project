# Car Insurance Claim Prediction Project

This project provides an end-to-end pipeline for modeling car insurance claim outcomes using machine learning. It demonstrates data preprocessing, feature engineering, model training, evaluation, and inference in a single notebook.

## Project Overview
The goal is to predict whether a car insurance policyholder will make a claim, using historical policy and driver information. The pipeline includes:
Handling numeric, categorical, and date features.
Feature engineering (e.g., domain-specific ratios).
Model selection and threshold tuning based on misclassification costs.
Evaluation with ROC-AUC, PR-AUC, Brier score, and slice-based metrics.
Optional SHAP explainability for model interpretation.

## Key Features
End-to-end machine learning workflow in a single Jupyter Notebook.
Automatic environment setup and package installation.
Domain-specific feature engineering.
Permutation importance and SHAP explainability support.
Threshold tuning using cost-sensitive metrics.
Slice-based error analysis to detect model weaknesses.
Predictive inference utility from CSV files.

## File structure
car_insurance_project/
│
├── notebooks/
│   └── insurance_claim_modeling_end_to_end.ipynb  # Main analysis and modeling notebook
│
├── data/
│   └── car_insurance.csv                          # Input dataset
│
├── README.md                                      # Project description and instructions

## Usage
1. Open the notebook in Jupyter or VSCode.
2. Update the data_path variable in the Configuration cell to point to ../data/car_insurance.csv.
3. Run all cells in order.
4. Model outputs and metrics will be generated inline in the notebook.

## Requirements
Python 3.8+ with pip.
The notebook auto-installs most missing packages if they are not already present.
