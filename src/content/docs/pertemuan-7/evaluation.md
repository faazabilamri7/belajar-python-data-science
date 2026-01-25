---
title: Model Training & Evaluation
description: Training models, evaluation metrics, overfitting, dan hyperparameter tuning
sidebar:
  order: 5
---

## 📊 Model Evaluation Metrics

Setelah training model, kita harus evaluate performance dengan metrics yang appropriate.

### Classification Metrics

#### 1. Accuracy

Persentase prediksi benar dari total.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Formula: (TP + TN) / (TP + TN + FP + FN)
# Dimana: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative
```

**Kapan pakai:**
- ✅ Balanced dataset
- ❌ Imbalanced dataset (misleading jika class distribution skewed)

#### 2. Precision & Recall

```
Precision: "Dari yang diprediksi positive, berapa yang benar?"
Formula: TP / (TP + FP)

Recall: "Dari yang benar-benar positive, berapa yang ketangkap?"
Formula: TP / (TP + FN)
```

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")  # Avoid false alarms
print(f"Recall: {recall:.4f}")        # Catch positives
print(f"F1-Score: {f1:.4f}")          # Harmonic mean
```

**Trade-off:**

```
High Precision, Low Recall: Conservative (few false alarms, miss many)
Low Precision, High Recall: Aggressive (catch many, many false alarms)
```

**Example Use Cases:**

| Problem | Priority | Metric |
| ------- | -------- | ------ |
| Medical diagnosis | Miss few cases | Recall (catch all diseases) |
| Spam detection | Avoid blocking emails | Precision (few false positives) |
| Credit fraud | Balanced | F1-score |

#### 3. Confusion Matrix

Detailed breakdown dari predictions.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

#### 4. ROC Curve & AUC

Visualize trade-off antara True Positive Rate dan False Positive Rate.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_pred_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"AUC Score: {roc_auc:.4f}")  # 1.0 = perfect, 0.5 = random
```

### Regression Metrics

#### Mean Squared Error (MSE)

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

# Formula: mean((y_actual - y_pred)^2)
```

#### Root Mean Squared Error (RMSE)

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Same scale sebagai target variable
```

#### Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

# Formula: mean(|y_actual - y_pred|)
# Less sensitive terhadap outliers daripada RMSE
```

#### R² Score

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Range: (-∞, 1.0]
# 1.0 = perfect, 0 = mean baseline, <0 = worse than mean
```

---

## 🔍 Overfitting & Underfitting

### Diagnosis

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='Validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# Analyze
plot_learning_curve(model, X, y)
```

### Diagnoses

```
UNDERFITTING:
- Training error: High
- Test error: High
- Gap: Small
- Solution: Use more complex model, more features

GOOD FIT:
- Training error: Low
- Test error: Low
- Gap: Small
✅ OPTIMAL

OVERFITTING:
- Training error: Very Low
- Test error: High
- Gap: Large
- Solution: Regularization, more data, simpler model
```

### Preventing Overfitting

#### 1. More Data

```python
# Collect more training data
# Model akan less "creative" dengan lebih banyak contoh
```

#### 2. Regularization

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier

# L2 Regularization (Ridge)
ridge = Ridge(alpha=1.0)  # Penalize large coefficients

# L1 Regularization (Lasso)
lasso = Lasso(alpha=1.0)  # Can shrink coefficients to 0

# Tree-based regularization
rf = RandomForestClassifier(
    max_depth=10,           # Limit tree depth
    min_samples_split=5,    # Require min samples to split
    min_samples_leaf=2      # Require min samples in leaf
)
```

#### 3. Early Stopping

```python
from sklearn.neural_network import MLPClassifier

# Stop training kalau validation error mulai increase
model = MLPClassifier(early_stopping=True, validation_fraction=0.1)
model.fit(X_train, y_train)
```

#### 4. Cross-Validation

Better estimate dari test performance dengan multiple folds.

```python
from sklearn.model_selection import cross_validate

# 5-fold cross validation
scores = cross_validate(model, X, y, cv=5, 
                       scoring=['accuracy', 'precision', 'recall'])

print(f"Accuracy: {scores['test_accuracy'].mean():.4f} "
      f"(+/- {scores['test_accuracy'].std():.4f})")
```

---

## 🎛️ Hyperparameter Tuning

Hyperparameter adalah setting model yang tidak belajar dari data, harus set manual.

### Grid Search

Coba semua kombinasi hyperparameter.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Use all processors
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

### Random Search

Coba kombinasi random (faster untuk large parameter space).

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,  # Try 20 random combinations
    cv=5,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

---

## 📈 Complete Training Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load & prepare
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Handle missing & encode
X = X.fillna(X.mean())
X = pd.get_dummies(X)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)

print("=== EVALUATION ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print("\nTop 10 Features:")
print(importance)
```

---

## ✏️ Latihan

### Latihan 1: Metrics Interpretation

Diberikan confusion matrix:
```
        Predicted
        Neg Pos
Actual Neg  85  5
       Pos  15  95
```

Hitung: Accuracy, Precision, Recall, F1-score. Interpret hasil!

### Latihan 2: Overfitting Diagnosis

Train 3 models:
1. Very simple (max_depth=1)
2. Medium (max_depth=5)
3. Complex (max_depth=20)

Plot learning curve untuk ketiga model. Identify mana yang underfitting, good fit, overfitting.

### Latihan 3: Hyperparameter Tuning

1. Use GridSearchCV untuk tune RandomForest parameters
2. Compare best model dengan default model
3. Check improvement dalam CV score dan test score

### Latihan 4: Complete Pipeline

End-to-end: Load → Prepare → Train → Evaluate dengan proper metrics dan visualizations.

---

## 🔗 Referensi

- [Scikit-Learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-Learn Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- [Understanding Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a-simple-way-52461ba0135f)
