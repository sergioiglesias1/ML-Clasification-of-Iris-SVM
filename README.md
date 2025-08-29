# ML-Classification-of-Iris-SVM

This project implements a comprehensive machine learning pipeline for classifying Iris flower species using **Support Vector Machines** (SVM) with **RBF kernel optimization**. The solution includes data exploration, preprocessing, hyperparameter tuning, and detailed performance evaluation.

## Features
- **Complete Data Analysis**: Exploratory data analysis with correlation matrices and feature distributions.
- **Preprocessing Pipeline**: StandardScaler integration for feature normalization.
- **SVM with RBF Kernel**: Implementation using the kernel function: K(xi ,xj)=exp(−γ∥xi−xj​∥2)
- **Hyperparameter Optimization**: Grid search for optimal C and γ parameters.
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix analysis.
- **Visualization Suite**: Multi-panel plots for data exploration and model interpretation.

## Quick Overview
The Iris dataset contains 150 samples equally distributed among three species:
- Setosa
- Versicolor
- Virginica
Each flower is described by four numerical features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Model Architecture
The classification pipeline consists of:
- **Data Standardization**: Features are normalized using StandardScaler.
- **SVM Classifier**: RBF kernel implementation for non-linear separation.
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation.
- **Optimal Parameters**: Search space includes:
	- C: [0.1, 1, 10, 100] (regularization parameter)
	-	gamma: [0.001, 0.01, 0.1, 1, 'scale'] (kernel coefficient)

## Performance Metrics
The model achieves high classification accuracy with detailed per-class metrics:
- Precision, recall, and F1-score for each species.
- Confusion matrix analysis.
- Cross-validation and test set accuracy comparison.

## Visualization
The project includes comprehensive visualizations:
- Correlation matrix of feature relationships.
- Distribution plots of petal and sepal measurements.
- Confusion matrix for model performance assessment.
- Comparative analysis of feature discriminative power.

## Key Findings
- Petal features are more discriminative than sepal features.
- Setosa is easily distinguishable from other species.
- Versicolor and Virginica show some feature overlap.
- The optimized SVM model achieves excellent classification performance.

## Tech Used
- **Python 3**
- **pandas**
- **NumPy**
- **scikit-learn**
- **Matplotlib & Seaborn**

## Project Structure
- ml_svm → Main notebook with complete analysis pipeline.
- Data loading, preprocessing, and exploration.
- Model training and hyperparameter optimization.
- Performance evaluation and visualization.
- Results interpretation and conclusions.

# Applications
- Classification systems.
- Pattern recognition.

# Usage
The notebook provides a complete walkthrough from data loading to model evaluation. Simply run the cells sequentially to reproduce the analysis or modify parameters for experimentation.
