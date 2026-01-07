# Titanic Survival Prediction

A machine learning project to predict passenger survival on the Titanic using ensemble methods and iterative feature engineering.

## Overview

This project analyzes the Titanic passenger dataset and builds classification models through an iterative approach, progressing from baseline ensembles to optimized feature selection and hyperparameter tuning.

## Dataset

- **Source**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- **Training set**: 891 passengers
- **Test set**: 418 passengers

## Approach & Results

### Round 1: Baseline Ensemble (Score: 0.79658)
- Established baseline using gradient boosting frameworks
- Models: XGBoost, CatBoost, LightGBM
- **Ensemble Method**: Hard voting
- **Feature Engineering**: Minimal/basic features
- **Version**: V6

### Round 2: Feature Engineering Focus (Score: 0.80173)
- Concentrated on feature engineering improvements
- Same model combination (XGBoost, CatBoost, LightGBM)
- **Ensemble Method**: Switched to soft voting
- **Improvement**: +0.00515 (+0.65%)
- **Version**: V3

### Round 3: Feature Selection & Optimization (Score: 0.80476)
- Used XGBoost for feature importance analysis
- Hyperparameter tuning with Optuna
- **Ensemble Method**: Soft voting
- **Improvement**: +0.00303 (+0.38%)
- **Version**: V3

## Key Techniques

- **Gradient Boosting Ensemble**: XGBoost, CatBoost, LightGBM
- **Voting Strategies**: Hard voting (R1) → Soft voting (R2-R3)
- **Feature Selection**: XGBoost feature importance
- **Hyperparameter Tuning**: Optuna optimization framework

## Final Performance

- **Best Kaggle Score**: 0.80476
- **Overall Improvement**: +0.00818 (+1.03% from baseline)
