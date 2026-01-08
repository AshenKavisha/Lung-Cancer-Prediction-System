# 🫁 Lung Cancer Prediction System

A machine learning-based classification system for predicting lung cancer risk using patient health data. This project implements an optimized feature engineering pipeline with advanced preprocessing techniques to achieve high-quality predictions for medical diagnosis support.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project develops a comprehensive machine learning pipeline for lung cancer prediction, focusing on:

- **Feature Engineering**: Optimized pipeline using SelectKBest and mutual information scoring
- **Dimensionality Reduction**: Reduces full feature set to top 10 most predictive features
- **Data Quality**: End-to-end preprocessing including class balancing, normalization, and correlation analysis
- **Reproducibility**: Fully automated Python pipeline for consistent results

The system is designed to assist medical professionals in early detection and risk assessment of lung cancer using patient health indicators and lifestyle factors.

## ✨ Features

### Core Capabilities
- **Automated Feature Selection**: SelectKBest with mutual information scoring to identify most predictive features
- **Class Balancing**: Handles imbalanced datasets for improved model performance
- **Data Normalization**: Standardizes features for optimal algorithm performance
- **Correlation Analysis**: Identifies and handles multicollinearity (threshold >0.85)
- **Visualization**: Comprehensive plots and heatmaps for data exploration

### Feature Engineering
Custom engineered features include:
- Age-Smoking Interaction
- Chronic Health Score
- Symptom Severity Index
- Age Risk Ratio
- Smoking-Alcohol Combined Impact
- Age-Symptom Index
- Chronic-Symptom Risk Score
- Breathing Difficulty Score
- Lifestyle Health Ratio

## 🛠️ Tech Stack

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Seaborn**: Statistical data visualization
- **Matplotlib**: Plotting and visualization

## 📁 Project Structure

```
lung-cancer-prediction/
│
├── Notebook/
│   └── Ashen.ipynb              # Main Jupyter notebook with complete pipeline
│
├── data/
│   └── raw/                     # Original dataset
│
├── results/
│   ├── outputs/                 # Processed datasets
│   │   ├── train_balanced_normalized.csv
│   │   ├── test_normalized.csv
│   │   ├── train_selected_features.csv
│   │   └── test_selected_features.csv
│   └── visualizations/          # Generated plots and heatmaps
│       └── correlation_matrix.png
│
└── README.md                    # Project documentation
```

## 🚀 Installation

### Prerequisites
Ensure you have Python 3.8+ installed on your system.

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lung-cancer-prediction.git
   cd lung-cancer-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib jupyter
   ```

   Or use a requirements file:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Pipeline

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**
   Navigate to `Notebook/Ashen.ipynb` and run all cells

3. **Expected outputs**
   - Preprocessed datasets in `results/outputs/`
   - Correlation matrix visualization in `results/visualizations/`
   - Feature selection results printed to console

### Code Example

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Load preprocessed data
train_df = pd.read_csv('results/outputs/train_balanced_normalized.csv')
test_df = pd.read_csv('results/outputs/test_normalized.csv')

# Separate features and target
X_train = train_df.drop(columns=['LUNG_CANCER'])
y_train = train_df['LUNG_CANCER']

# Apply feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=10)
selector.fit(X_train, y_train)

# Get selected features
selected_features = X_train.columns[selector.get_support()].tolist()
print("Top 10 predictive features:", selected_features)
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Loading**: Import raw patient health data
- **Cleaning**: Handle missing values and outliers
- **Encoding**: Convert categorical variables to numerical format

### 2. Feature Engineering
- Create interaction terms between related features
- Develop composite health scores
- Calculate risk ratios and severity indices

### 3. Class Balancing
- Address imbalanced target variable distribution
- Ensure equal representation of positive and negative cases

### 4. Normalization
- Apply min-max scaling or standardization
- Ensure all features are on comparable scales

### 5. Correlation Analysis
- Generate correlation matrix with visualization
- Identify highly correlated features (threshold: 0.85)
- Flag multicollinearity issues

### 6. Feature Selection
- Use SelectKBest with mutual information scoring
- Rank features by predictive power
- Select top 10 features for modeling

### 7. Output Generation
- Save processed training and test datasets
- Export selected feature datasets
- Generate visualization artifacts

## 📊 Results

### Feature Selection Output
The pipeline identifies the top 10 most predictive features using mutual information scoring, reducing dimensionality while maintaining predictive power.

### Correlation Analysis
- Correlation matrix visualization saved to `results/visualizations/correlation_matrix.png`
- Identifies feature pairs with correlation >0.85
- Helps prevent multicollinearity in downstream models

### Data Quality Improvements
- Balanced class distribution for training
- Normalized feature scales
- Optimized feature set for prediction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Steps to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

**AshenKavisha**

- GitHub: [@AshenKavisha](https://github.com/AshenKavisha)

## 🙏 Acknowledgments

- Dataset source and medical domain experts
- Open-source community for tools and libraries
- Contributors and collaborators


