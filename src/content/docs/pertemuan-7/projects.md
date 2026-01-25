---
title: Real-World Projects
description: End-to-end ML projects dan best practices
sidebar:
  order: 6
---

## 🎯 Project 1: Customer Churn Prediction

Predict apakah customer akan churn (leave) atau tidak.

### Problem Definition

```python
# Problem: Binary classification
# Task: Predict jika customer churn atau tidak
# Metric: F1-score (balanced precision & recall)
# Target: Churn (Yes/No)
```

### Complete Solution

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA
df = pd.read_csv('customer_churn.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nTarget distribution:")
print(df['Churn'].value_counts())

# 2. EXPLORATORY ANALYSIS
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Churn distribution
df['Churn'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Churn Distribution')

# Age vs Churn
df.boxplot(column='Age', by='Churn', ax=axes[0, 1])
axes[0, 1].set_title('Age by Churn')

# Tenure vs Churn
sns.scatterplot(data=df, x='Tenure', y='MonthlyCharges', hue='Churn', ax=axes[1, 0])
axes[1, 0].set_title('Tenure vs Monthly Charges')

# Correlation heatmap
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix')

plt.tight_layout()
plt.show()

# 3. DATA PREPARATION

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode target
le_target = LabelEncoder()
df['Churn'] = le_target.fit_transform(df['Churn'])

# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Churn')  # Remove target

# Encode categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 4. SPLIT DATA
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 5. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. MODEL TRAINING

# Grid search untuk optimal parameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# 7. MODEL EVALUATION
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("\n=== EVALUATION METRICS ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'])
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.show()

print("\nTop 10 Features:")
print(importance)
```

---

## 🎯 Project 2: House Price Prediction

Predict harga rumah berdasarkan features.

### Problem Definition

```python
# Problem: Regression
# Task: Predict house price
# Metric: RMSE, R² score
# Target: Price
```

### Solution

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('house_prices.csv')

# 2. Data preparation
# Drop missing values > 50%
df = df.dropna(thresh=len(df)*0.5, axis=1)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. Split data
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:,.0f}")
print(f"MAE: ${mae:,.0f}")
print(f"R² Score: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                            scoring='r2')
print(f"CV R² Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 7. Visualize results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title('Actual vs Predicted')

# Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Predicted Price')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')

plt.tight_layout()
plt.show()
```

---

## 🎯 Project 3: Multi-class Classification

Classify iris flowers ke 3 species.

### Solution

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# 1. Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\nCross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## 📋 Best Practices Checklist

### Before Training

- [ ] Define problem clearly (regression/classification/clustering)
- [ ] Explore data thoroughly (EDA)
- [ ] Handle missing values appropriately
- [ ] Encode categorical variables
- [ ] Split data (train/test) BEFORE preprocessing
- [ ] Scale numeric features
- [ ] Address class imbalance (if applicable)

### During Training

- [ ] Start with simple model as baseline
- [ ] Use cross-validation untuk better estimation
- [ ] Monitor for overfitting/underfitting
- [ ] Tune hyperparameters systematically
- [ ] Use stratified split untuk classification

### After Training

- [ ] Evaluate dengan multiple metrics (appropriate untuk problem type)
- [ ] Analyze errors dan misclassifications
- [ ] Check feature importance
- [ ] Validate pada completely separate test set
- [ ] Document model performance dan decisions

### General

- [ ] Use consistent random_state untuk reproducibility
- [ ] Keep code organized dan commented
- [ ] Version control code dan models
- [ ] Document assumptions dan limitations
- [ ] Consider computational resources

---

## ✏️ Challenge Exercises

### Challenge 1: Kaggle Dataset

1. Pick dataset dari Kaggle
2. Complete end-to-end ML pipeline
3. Achieve >80% accuracy (atau appropriate R² for regression)
4. Document entire process

### Challenge 2: Imbalanced Data

1. Find imbalanced dataset (e.g., fraud detection)
2. Try 3 approaches: resampling, class weights, different metrics
3. Compare performance
4. Document findings

### Challenge 3: Production Pipeline

1. Build complete ML pipeline (load → prepare → train → evaluate)
2. Save model dengan joblib
3. Create prediction function untuk new data
4. Test on unseen data

### Challenge 4: Hyperparameter Optimization

1. Pick complex model
2. Use GridSearchCV / RandomizedSearchCV
3. Document best parameters found
4. Compare performance: default vs tuned
5. Calculate improvement percentage

---

## 🔗 Referensi

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-Learn Examples](https://scikit-learn.org/stable/auto_examples/index.html)
- [Applied Machine Learning](https://machinelearningmastery.com/)
- [Real-World ML Projects](https://github.com/topics/machine-learning-project)
