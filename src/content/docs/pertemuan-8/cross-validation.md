---
title: Cross-Validation Techniques
description: K-Fold CV, stratified splitting, dan reliable performance estimation
sidebar:
  order: 5
---

## 🎯 Why Cross-Validation?

### Problem dengan Single Train-Test Split

```python
# Problem: Results dapat berbeda-beda
# Tergantung random split yang dipilih

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)
model1.fit(X_train1, y_train1)
score1 = model1.score(X_test1, y_test1)  # 0.87

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=123)
model2.fit(X_train2, y_train2)
score2 = model2.score(X_test2, y_test2)  # 0.82

# Score beda! Mana yang reliable?
```

### Solution: Cross-Validation

Cross-validation menggunakan **multiple train-test splits** dan average hasilnya untuk mendapatkan **robust estimate** dari true performance.

```
Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Fold 1: Train=[3-10], Test=[1-2]    → Score: 0.85
Fold 2: Train=[1,2,5-10], Test=[3-4]  → Score: 0.87
Fold 3: Train=[1-4,7-10], Test=[5-6]  → Score: 0.83
Fold 4: Train=[1-6,9-10], Test=[7-8]  → Score: 0.88
Fold 5: Train=[1-8], Test=[9-10]     → Score: 0.86

Final Score = Average = 0.858 ± 0.019
            = Better estimate!
```

---

## 📊 K-Fold Cross-Validation

### Basic Implementation

```python
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold CV dengan scoring='accuracy'
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV Scores: {scores}")
# Output: [0.85 0.87 0.83 0.88 0.86]

print(f"Mean: {scores.mean():.4f}")
print(f"Std:  {scores.std():.4f}")
# Output: Mean: 0.8580, Std: 0.0181

# Interpretation:
# Score adalah 0.858 ± 0.018 (95% confidence)
```

### Choosing K

| K Value | Pros | Cons |
| ------- | ---- | ---- |
| 3 | Fast | High variance |
| **5** | **Good balance** | **Recommended** |
| 10 | Lower variance | Slower |
| n (Leave-One-Out) | Lowest variance | Very slow |

```python
# Rekomendasi
cv=5      # Default untuk most cases

# Jika dataset kecil
cv=3      # Faster

# Jika dataset besar & time is not concern
cv=10     # More reliable
```

### Multiple Metrics

```python
from sklearn.model_selection import cross_validate

# Define multiple metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# Cross validate
results = cross_validate(model, X, y, cv=5, scoring=scoring)

# Results untuk setiap fold
print(results.keys())
# dict_keys(['fit_time', 'score_time', 'test_accuracy', 'test_precision', ...])

# Average hasil
for metric in scoring.keys():
    scores = results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 🎯 Stratified K-Fold

### Problem: Imbalanced Dataset

```python
# Imbalanced dataset
print(y.value_counts())
# 0    950  (95%)
# 1     50  (5%)

# Regular K-Fold bisa menghasilkan fold yang tidak representative
# Contoh: Fold 1 bisa punya 100% positive samples
# → Unrealistic scenario

# Solution: Stratified K-Fold
```

### Implementation

```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold maintains class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

print(f"Stratified CV Scores: {scores}")
# Setiap fold akan punya 95% class 0, 5% class 1
# → More representative!
```

### Automatic Stratification

```python
# Dengan stratify parameter di train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y  # ← Maintains ratio
)

# Dengan stratify di cross_val_score
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=cv)
```

---

## 🔄 Leave-One-Out Cross-Validation (LOOCV)

### Concept

LOOCV adalah extreme case dari K-Fold dimana K = n (jumlah samples).

```
n = 100 samples
LOOCV:
- Fold 1: Train=99, Test=1
- Fold 2: Train=99, Test=1
- ...
- Fold 100: Train=99, Test=1

Average dari 100 folds = Unbiased estimate
```

### Implementation

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)

print(f"LOOCV Score: {scores.mean():.4f}")
```

### Kapan Menggunakan LOOCV?

- ✅ Very small dataset (n < 50)
- ✅ Need most reliable estimate
- ❌ Computationally expensive
- ❌ High variance untuk large datasets

---

## 🎨 Time Series Cross-Validation

### Problem: Regular CV tidak cocok untuk time series

```
Regular K-Fold bisa menghasilkan:
Train: [future data], Test: [past data]
→ Unrealistic (model menggunakan future data)

Solution: Time Series CV (forward chaining)
```

### Implementation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

# Setiap split: train pada past data, test pada future data
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Fold score: {score:.4f}")
```

---

## 📊 Cross-Validation in Practice

### Complete Example

```python
from sklearn.model_selection import (cross_val_score, 
                                     cross_validate,
                                     StratifiedKFold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Setup
model = RandomForestClassifier(n_estimators=100, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Method 1: Single metric
print("Method 1: Single Metric")
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Method 2: Multiple metrics
print("\nMethod 2: Multiple Metrics")
scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'precision': 'precision'}
results = cross_validate(model, X, y, cv=skf, scoring=scoring)

for metric in scoring.keys():
    scores = results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Method 3: Manual (untuk kontrol penuh)
print("\nMethod 3: Manual CV Loop")
fold_scores = []
for i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    fold_scores.append(score)
    
    print(f"Fold {i}: {score:.4f}")

print(f"Mean: {np.mean(fold_scores):.4f}, Std: {np.std(fold_scores):.4f}")
```

---

## 🎯 Cross-Validation Tips

### Tip 1: Stratified untuk Classification

```python
# ✅ CORRECT untuk imbalanced classification
cv = StratifiedKFold(n_splits=5)

# ❌ Dapat problematic untuk imbalanced
cv = KFold(n_splits=5)
```

### Tip 2: Scale Features INSIDE CV

```python
# ❌ WRONG: Fit scaler pada semua data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(model, X_scaled, y, cv=5)
# → Data leakage! (test data influence training)

# ✅ CORRECT: Use Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
scores = cross_val_score(pipeline, X, y, cv=5)
# → No leakage! Scaler fit pada train data saja
```

### Tip 3: Random State untuk Reproducibility

```python
# Tanpa random_state
cv = StratifiedKFold(n_splits=5)
# Setiap run berbeda folds

# Dengan random_state
cv = StratifiedKFold(n_splits=5, random_state=42)
# Setiap run same folds
```

---

## 📊 CV Score Interpretation

### What Do Numbers Mean?

```
scores = [0.85, 0.87, 0.83, 0.88, 0.86]

Mean = 0.858
Std = 0.0181

Interpretation:
- Expected performance: 85.8%
- Variance: ±1.81% (95% confidence interval)
- Range: 83% - 88% (across folds)

Jika 1 fold lebih rendah (83%):
→ OK, bisa terjadi karena randomness

Jika semua folds berbeda jauh:
→ Model unstable, butuh improvement
```

---

## 🔍 Debugging Low CV Scores

```python
# Problem: CV scores rendah

# Check 1: Data quality
print(df.isnull().sum())        # Missing values?
print(df.describe())             # Outliers?

# Check 2: Class imbalance
print(y.value_counts())          # Balanced?

# Check 3: Feature quality
print(df.corr())                 # Correlations?

# Check 4: Model complexity
# Try simpler model → CV score naik?
# → Overfitting!

# Check 5: Hyperparameters
# Try different parameters → CV score naik?
# → Tuning needed!
```

---

## ✏️ Latihan

### Latihan 1: Basic CV

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# 1. Load iris dataset
# 2. Train RandomForestClassifier dengan 5-Fold CV
# 3. Print mean & std dari CV scores
# 4. Interpret hasil
```

### Latihan 2: Stratified vs Regular

```python
from sklearn.model_selection import KFold, StratifiedKFold

# Pada imbalanced dataset:
# 1. Compare KFold vs StratifiedKFold
# 2. Which one gives more stable scores?
# 3. Which one more realistic for imbalanced data?
```

### Latihan 3: Multiple Metrics

```python
# 1. Define 5 different metrics
# 2. Perform cross_validate dengan 5-Fold CV
# 3. Compare scores untuk setiap metric
# 4. Which metric most important untuk problem kamu?
```

### Latihan 4: Pipeline with CV

```python
from sklearn.pipeline import Pipeline

# 1. Create pipeline: Scaler → Model
# 2. Perform cross_val_score pada pipeline
# 3. Verify no data leakage
# 4. Compare dengan manual scaling approach
```

---

## 🔗 Referensi

- [Scikit-Learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Time Series CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Nested CV for Hyperparameter Tuning](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
