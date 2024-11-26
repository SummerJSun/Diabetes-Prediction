# Diabetes Probability Prediction Using Machine Learning

This project aims to predict the probability of diabetes based on multiple risk factors using various machine learning algorithms. We analyze data from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) to assess diabetes risk through both clinical and socioeconomic factors.

## Overview

Diabetes affects over 34 million individuals in the United States, with approximately 88 million Americans living with prediabetes. Many are unaware of their condition. This project leverages machine learning to:
- Predict the probability of individuals developing diabetes
- Identify the most significant indicators contributing to diabetes risk
- Compare the predictive power of social vs. clinical features

## Dataset

The dataset is derived from the CDC's 2015 telephone survey results, containing:
- 253,680 responses
- 21 critical indicators for diabetes risk
- Features including high blood pressure, BMI, smoking habits, and heart disease
- All features cleaned and converted to numerical types
- Binary variables one-hot encoded

## Methods

Our analysis pipeline includes:
1. **Data Preprocessing**
   - Dataset balancing
   - Feature scaling
   - Feature selection using Pearson correlation and mutual information

2. **Model Implementation**
   - XGBoost Classifier
   - CatBoost
   - Decision Tree
   - Logistic Regression
   - K-Nearest Neighbors
   - Neural Network (MLPClassifier)
   - GaussianNB
   - Random Forest Classifier
   - Stacking

3. **Feature Selection**
   - Regular approach using correlation and mutual information
   - Random Forest-based feature importance ranking

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1 Score
   - ROC AUC and AUPRC
   - Calibration curves

## Key Findings

- CatBoost demonstrated the highest overall performance with:
  - Accuracy: 0.7576
  - F1 Score: 0.7655
  - AUC: 0.8358

- Top 5 important features:
  - General Health
  - BMI
  - Age
  - High Blood Pressure
  - High Cholesterol

- Clinical features showed stronger predictive power than social factors, but combining both feature sets yielded the best results

## Contributors

- Wenye Song
- Jinghan Sun
- Iris Zheng

## Resources

- [Google Colab Notebook](https://colab.research.google.com/drive/1phLBmwR03ywWcklvnBe8WdM0RF4gTL8r#scrollTo=8p5nojgpo6l0&line=1&uniqifier=1)
- [Dataset](https://drive.google.com/file/d/1vel7TodXYvnw6_JW49x_h9le9BkmyBrO/view?usp=sharing)

## References

1. A. Teboul, "Diabetes Health Indicators Dataset," Kaggle, 2021.
2. Z. Xie, O. Nikolayeva, J. Luo, and D. Li, "Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques," Prev Chronic Dis, 2019.
3. I. Tasin, T. U. Nabil, S. Islam, and R. Khan, "Diabetes prediction using machine learning and explainable AI techniques," Healthc Technol Lett, 2023.

