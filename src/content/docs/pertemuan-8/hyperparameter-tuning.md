---
title: Hyperparameter Tuning
description: GridSearchCV, RandomizedSearchCV, dan parameter optimization
sidebar:
  order: 6
---

## 🎯 What are Hyperparameters?

### Parameters vs Hyperparameters

```
PARAMETERS:
- Learned dari data during training
- Weights & biases in neural networks
- Coefficients in linear models
- Can't manually set

HYPERPARAMETERS:
- Set sebelum training
- Control bagaimana model belajar
- Manually tuned / searched
- Examples: n_estimators, max_depth, learning_rate
```

### Why Tune Hyperparameters?

```
Default hyperparameters:
model = RandomForestClassifier()  # Accuracy = 0.82

Tuned hyperparameters:
model = RandomForestClassifier(n_estimators=500, max_depth=10, ...)
# Accuracy = 0.87 ← 5% improvement!
```

---

## 🔍 Grid Search CV

### Basic Concept

Grid Search menggunakan **brute-force approach**: coba semua kombinasi hyperparameter yang diberikan.

```
Parameters:
n_estimators: [50, 100, 200]
max_depth: [5, 10, 20]

Grid:
[50, 5], [50, 10], [50, 20]
[100, 5], [100, 10], [100, 20]
[200, 5], [200, 10], [200, 20]

Total kombinasi: 3 × 3 = 9
Training & evaluation untuk masing-masing.
```

### Implementation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create base model
base_model = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,                 # 5-Fold CV
    scoring='accuracy',   # Metric to optimize
    n_jobs=-1            # Use all processors
)

# Fit (this will try all combinations)
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

### Exploring Results

```python
# Get all results
results_df = pd.DataFrame(grid_search.cv_results_)

print(results_df[['param_n_estimators', 'param_max_depth', 
                  'mean_test_score', 'std_test_score']])

# Plot scores vs parameters
import matplotlib.pyplot as plt

scores = results_df['mean_test_score'].values
n_est = results_df['param_n_estimators'].values

plt.scatter(n_est, scores)
plt.xlabel('n_estimators')
plt.ylabel('Mean CV Score')
plt.show()
```

---

## 🎲 Random Search CV

### Concept

Random Search mencoba **kombinasi random** dari parameter space. Lebih cepat dari Grid Search, terutama untuk large parameter space.

```
Grid Search:  Try all 100 combinations (slow)
Random Search: Try 20 random combinations (fast)

Random Search sering menemukan kombinasi yang baik
dengan jauh lebih cepat!
```

### Implementation

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'learning_rate': uniform(0.001, 0.1)
}

# Random Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,          # Coba 50 kombinasi random
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

### Grid vs Random

```
Small parameter space (≤10 parameters):
→ Use GridSearchCV (thorough)

Large parameter space (>10 parameters):
→ Use RandomizedSearchCV (faster)

Hybrid approach:
1. RandomizedSearchCV untuk narrow down (fast)
2. GridSearchCV untuk fine-tuning (thorough)
```

---

## 🔧 Common Hyperparameters to Tune

### RandomForest

```python
param_grid = {
    'n_estimators': [50, 100, 200, 500],      # Banyak trees
    'max_depth': [5, 10, 15, 20, None],        # Kedalaman
    'min_samples_split': [2, 5, 10, 20],       # Min samples untuk split
    'min_samples_leaf': [1, 2, 4, 8],          # Min samples di leaf
    'max_features': ['sqrt', 'log2']           # Features per split
}
```

### SVM

```python
param_grid = {
    'C': [0.1, 1, 10, 100],                   # Regularization
    'kernel': ['linear', 'rbf', 'poly'],      # Kernel type
    'gamma': ['scale', 'auto', 0.001, 0.01]   # Kernel coefficient
}
```

### Neural Network

```python
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'alpha': [0.0001, 0.001]  # L2 regularization
}
```

---

## 📊 Complete Hyperparameter Tuning Example

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("="*60)
print("HYPERPARAMETER TUNING: RandomForest on Iris")
print("="*60)

# Baseline
print("\n1️⃣ BASELINE (Default Hyperparameters)")
baseline = RandomForestClassifier(random_state=42)
baseline_scores = cross_val_score(baseline, X_train, y_train, cv=5)
print(f"Baseline CV Score: {baseline_scores.mean():.4f}")

# Grid Search
print("\n2️⃣ GRID SEARCH")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1  # Print progress
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Improvement
improvement = (grid_search.best_score_ - baseline_scores.mean()) * 100
print(f"Improvement: {improvement:+.2f}%")

# Test set performance
print("\n3️⃣ TEST SET PERFORMANCE")
baseline_test_score = baseline.fit(X_train, y_train).score(X_test, y_test)
tuned_test_score = grid_search.best_estimator_.score(X_test, y_test)

print(f"Baseline Test Score: {baseline_test_score:.4f}")
print(f"Tuned Test Score: {tuned_test_score:.4f}")
print(f"Improvement: {(tuned_test_score - baseline_test_score)*100:+.2f}%")

# Visualize results
print("\n4️⃣ RESULTS SUMMARY")
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('mean_test_score', ascending=False)
print(results_df[['param_n_estimators', 'param_max_depth', 
                  'param_min_samples_split', 'mean_test_score', 'std_test_score']].head(10))
```

---

## 🎯 Best Practices

### Tip 1: Reasonable Parameter Range

```python
# ❌ TOO WIDE
param_grid = {
    'n_estimators': [1, 1000, 10000],  # Sparse coverage
}

# ✅ REASONABLE
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],  # Dense coverage
}
```

### Tip 2: Use Cross-Validation Inside Grid Search

```python
# ✅ CORRECT: GridSearchCV sudah gunakan CV
grid_search = GridSearchCV(model, param_grid, cv=5)

# ❌ WRONG: Jangan split manual
X_train, X_test = train_test_split(X)
grid_search = GridSearchCV(model, param_grid)
grid_search.fit(X_train)  # CV hanya pada X_train saja!
```

### Tip 3: Use Pipeline untuk Preprocessing

```python
from sklearn.pipeline import Pipeline

# ✅ CORRECT: No data leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

grid_search = GridSearchCV(
    pipeline,
    {'model__n_estimators': [50, 100, 200]},  # ← Prefix dengan model__
    cv=5
)

# ❌ WRONG: Data leakage
X_scaled = scaler.fit_transform(X)  # Fit pada all data!
grid_search.fit(X_scaled, y)
```

### Tip 4: Nested CV untuk Unbiased Estimate

```python
from sklearn.model_selection import cross_val_score

# Outer CV untuk evaluate final model
outer_scores = cross_val_score(
    grid_search,  # GridSearchCV object
    X, y,
    cv=5
)

print(f"Unbiased estimate: {outer_scores.mean():.4f}")
```

---

## 🔍 Common Pitfalls

### Pitfall 1: Overfitting pada Test Set

```python
# ❌ WRONG
param_grid = {'C': [0.1, 1, 10, 100, 1000]}
grid_search.fit(X_train, y_train)
# Evaluate pada test set
score = grid_search.score(X_test, y_test)
# Problem: Grid Search sudah "see" X_test melalui best_params_!

# ✅ CORRECT
# GridSearchCV sudah handle ini dengan nested CV
```

### Pitfall 2: Tuning Terlalu Banyak Parameters

```python
# ❌ WRONG
param_grid = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_depth': [1, 2, 3, 4, 5, 10, 15, 20],
    'min_samples_split': [2, 3, 4, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}
# Total: 5 × 8 × 5 × 4 × 2 × 2 = 3200 kombinasi!

# ✅ CORRECT
# Prioritas: tune parameters paling sensitive dulu
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20]
}
```

---

## ✏️ Latihan

### Latihan 1: Basic Grid Search

```python
# 1. Load Titanic dataset
# 2. Define param_grid dengan 3 parameters
# 3. Perform GridSearchCV dengan 5-Fold CV
# 4. Print best parameters dan best score
# 5. Compare dengan baseline
```

### Latihan 2: Random Search

```python
# 1. Gunakan parameter space yang lebih besar
# 2. Perform RandomizedSearchCV (50 iterations)
# 3. Compare dengan GridSearchCV dalam hal:
#    - Time taken
#    - Best score found
```

### Latihan 3: Pipeline Tuning

```python
# 1. Create pipeline: Scaler + Model
# 2. Tune parameters: scaler dan model parameters
# 3. Use GridSearchCV pada pipeline
# 4. Verify no data leakage
```

### Latihan 4: Nested CV

```python
# 1. Create GridSearchCV object
# 2. Use cross_val_score sebagai outer CV
# 3. Get unbiased performance estimate
# 4. Compare dengan single CV score
```

---

## 🔗 Referensi

- [Scikit-Learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-Learn RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [Hyperparameter Tuning Best Practices](https://towardsdatascience.com/hyperparameter-tuning-in-machine-learning-26d0eccd6e2c)
