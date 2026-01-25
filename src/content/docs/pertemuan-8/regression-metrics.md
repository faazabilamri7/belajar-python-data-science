---
title: Regression Metrics & Evaluation
description: Metrik evaluasi untuk regression problems
sidebar:
  order: 4
---

## 📊 Regression Metrics Overview

Untuk regression problems (prediksi nilai kontinyu), kita punya berbagai metrics untuk mengukur seberapa dekat prediksi dengan nilai actual.

**Core Concept:** Regression mengukur magnitude dari error, bukan hanya benar/salah seperti classification.

```
Actual value: 100
Prediction 1: 105  (error = 5) ✓ Lebih baik
Prediction 2: 110  (error = 10) ✗ Lebih buruk

Classification hanya lihat benar/salah
Regression peduli berapa jauh errornya
```

---

## 📏 Mean Absolute Error (MAE)

**Definition:** Rata-rata absolute error dari semua predictions.

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

"Rata-rata, prediksi kita meleset berapa?

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

# Interpretation:
# Jika memprediksi harga rumah: MAE = 50,000
# Berarti rata-rata prediksi meleset Rp 50 juta
```

### Kelebihan & Kekurangan

**Kelebihan:**
- ✅ Mudah diinterpretasi (satuan sama dengan target)
- ✅ Robust terhadap outliers

**Kekurangan:**
- ❌ Tidak memberi penalti lebih pada error besar

---

## 📈 Mean Squared Error (MSE)

**Definition:** Rata-rata squared error.

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Dengan mengkuadratkan error, MSE memberi **penalti lebih besar** untuk error yang besar.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

# Interpretation:
# Rata-rata squared deviation dari predictions
# Nilai lebih besar = lebih banyak error besar
```

### Comparison: MAE vs MSE

```
Actual:      [100, 100, 100, 100]
Prediction A: [90, 110, 90, 110]   (errors: 10, 10, 10, 10)
Prediction B: [85, 115, 85, 115]   (errors: 15, 15, 15, 15)

MAE A = (10+10+10+10)/4 = 10
MAE B = (15+15+15+15)/4 = 15
→ B worse (correctly shows larger error)

MSE A = (100+100+100+100)/4 = 100
MSE B = (225+225+225+225)/4 = 225
→ B much worse (squared penalizes larger errors more)
```

### Kapan Menggunakan MSE?

- ✅ Ketika error besar lebih problematic
- ✅ Optimization problems (easier to differentiate)
- ❌ Harder to interpret (units are squared)

---

## 🔢 Root Mean Squared Error (RMSE)

**Definition:** Akar dari MSE.

$$RMSE = \sqrt{MSE}$$

RMSE "un-squares" MSE sehingga satuan sama dengan target variable. Ini membuat interpretasi lebih mudah sambil tetap memberi penalti pada error besar.

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Atau dengan scikit-learn
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

### Comparison: MAE vs MSE vs RMSE

```
Predictions vs Actual untuk housing prices:

Actual:      [200K, 300K, 400K]
Prediction:  [190K, 310K, 380K]
Errors:      [10K,  10K,  20K]

MAE = (10+10+20)/3 = 13,333
MSE = (100M + 100M + 400M)/3 = 200M
RMSE = √200M = 14,142

Interpretation:
- MAE: Rata-rata error Rp 13 juta
- RMSE: Rata-rata squared error 200M, atau root = Rp 14 juta
- RMSE sedikit lebih tinggi karena penalti pada error 20K
```

---

## 📊 R² Score (Coefficient of Determination)

**Definition:** Proportion of variance yang dijelaskan model.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Dimana:
- $SS_{res}$ = Sum of Squares Residual (actual - predicted)²
- $SS_{tot}$ = Total Sum of Squares (actual - mean)²

R² menunjukkan **berapa persen variance dalam target yang dijelaskan model**.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Interpretation:
# R² = 0.85 berarti 85% variance dalam target dijelaskan model
# 15% variance masih unexplained
```

### Interpretasi R²

| R² | Interpretasi |
| --- | ------------ |
| 1.0 | Perfect prediction |
| 0.9 | Excellent |
| 0.7 | Good |
| 0.5 | Moderate |
| 0.3 | Poor |
| 0.0 | Sama dengan prediksi mean (baseline) |
| < 0 | Lebih buruk daripada prediksi mean! |

### Example: R² Interpretation

```python
# Prediksi harga rumah
actual_prices = [100K, 150K, 200K, 250K, 300K]
mean_price = 200K  # Baseline: selalu prediksi mean

# Model A: memprediksi [102K, 148K, 202K, 248K, 298K]
# R² = 0.95 berarti model 95% lebih baik dari baseline

# Model B: memprediksi [199K, 201K, 200K, 200K, 200K]
# R² = 0.0 berarti model sama saja dengan prediksi mean

# Model C: memprediksi [50K, 350K, 100K, 400K, 200K]
# R² = -0.5 berarti model lebih buruk dari baseline!
```

---

## 🏆 Choosing the Right Metric

### Decision Tree

```
What matters?
│
├─ Small errors lebih baik? (magnitude)
│  └─ Use MAE
│
├─ Large errors harus dihindari?
│  └─ Use MSE atau RMSE
│
├─ Berapa persen variance yang explained?
│  └─ Use R²
│
└─ Multi-metric evaluation?
   └─ Use MAE + RMSE + R² bersama
```

### Real-World Examples

```
STOCK PRICE PREDICTION:
- Large errors very costly
- → Use RMSE atau MSE
- "Predictions yang meleset banyak sangat buruk"

WEATHER TEMPERATURE PREDICTION:
- Small errors acceptable
- → Use MAE
- "Rata-rata berapa degrees error adalah OK"

HOUSE PRICE PREDICTION:
- Need to explain model performance to stakeholders
- → Use R² (85% of price variance explained)
- "Model explains 85% dari variasi harga rumah"
```

---

## 📋 Adjusted R²

**Problem:** R² selalu naik ketika menambah features, bahkan jika features tidak helpful.

**Solution:** Adjusted R² memberikan penalti untuk features yang tidak helpful.

$$Adjusted\;R^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Dimana:
- $n$ = number of samples
- $p$ = number of features

```python
# Scikit-learn tidak punya built-in, tapi bisa dihitung manual
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2(r2, len(y_test), X_test.shape[1])

print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
```

### R² vs Adjusted R²

```
Scenario: Add 100 random features

Model A (10 features):
- R² = 0.85, Adjusted R² = 0.84

Model B (110 features):
- R² = 0.87  ← R² naik (tapi features random!)
- Adjusted R² = 0.65  ← Adjusted R² turun (correctly penalizes)

→ Adjusted R² lebih robust untuk model comparison
```

---

## 📊 Complete Regression Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print summary
print("="*50)
print("COMPREHENSIVE REGRESSION EVALUATION")
print("="*50)
print(f"Mean Absolute Error:  {mae:,.2f}")
print(f"Mean Squared Error:   {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"R² Score:             {r2:.4f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Actual vs Predicted')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals
residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals Distribution
axes[1, 0].hist(residuals, bins=30, edgecolor='black')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residuals Distribution')
axes[1, 0].axvline(x=0, color='r', linestyle='--')

# 4. Error Distribution
errors = np.abs(y_test - y_pred)
axes[1, 1].hist(errors, bins=30, edgecolor='black')
axes[1, 1].set_xlabel('Absolute Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Error Distribution (MAE={mae:.2f})')

plt.tight_layout()
plt.show()
```

---

## 🔍 Regression Diagnostics

### Residual Analysis

```python
residuals = y_test - y_pred

# Check if residuals are normally distributed
from scipy import stats

# Shapiro-Wilk test (p > 0.05 = normally distributed)
stat, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk test p-value: {p_value:.4f}")

if p_value > 0.05:
    print("✅ Residuals are normally distributed")
else:
    print("⚠️ Residuals are NOT normally distributed")

# Q-Q plot untuk visualisasi
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()
```

---

## ✏️ Latihan

### Latihan 1: Calculate Metrics

```python
# Actual prices: [100, 150, 200, 250, 300]
# Predicted:     [105, 145, 210, 245, 310]

# Calculate:
# 1. MAE
# 2. MSE
# 3. RMSE
# 4. R²
```

### Latihan 2: House Price Prediction

```python
from sklearn.datasets import fetch_california_housing

# 1. Load California housing dataset
# 2. Split train-test
# 3. Train LinearRegression model
# 4. Calculate MAE, MSE, RMSE, R²
# 5. Create visualizations (actual vs predicted, residuals)
# 6. Interpret results
```

### Latihan 3: Model Comparison

```python
# Train 3 different regression models
# Evaluate dengan semua metrics
# Create comparison table
# Recommend best model
```

---

## 🔗 Referensi

- [Scikit-Learn Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [R² Score Explained](https://towardsdatascience.com/r-squared-explained-dd6986f75b98)
- [Understanding Regression Metrics](https://medium.com/@koushikjanartha/performance-metrics-for-regression-problems-7dfeda5d3c1c)
