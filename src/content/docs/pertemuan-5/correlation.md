---
title: Korelasi & Hubungan Antar Variabel
description: Pearson, Spearman, Kendall dan visualisasi korelasi
sidebar:
  order: 6
---

## 🔗 Apa itu Korelasi?

**Korelasi** mengukur **hubungan linear antara dua variabel**. Apakah ketika satu variabel naik, yang lain juga naik? atau malah turun?

**Rentang nilai korelasi: -1 sampai +1**
- **r = +1**: Perfect positive correlation (keduanya naik bersama)
- **r = 0**: No correlation (independent)
- **r = -1**: Perfect negative correlation (satu naik, satu turun)

$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

---

## 📊 1. Pearson Correlation Coefficient

**Pearson r** adalah ukuran korelasi linear paling umum. Untuk data continuous dan **assume normal distribution**.

### Hitung Pearson Correlation

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate correlated data
x = np.random.randn(100)
y_positive = x * 2 + np.random.randn(100) * 0.5  # Strong positive correlation
y_negative = -x * 2 + np.random.randn(100) * 0.5  # Strong negative correlation
y_none = np.random.randn(100)  # No correlation

# Method 1: SciPy
r_pos, p_pos = stats.pearsonr(x, y_positive)
r_neg, p_neg = stats.pearsonr(x, y_negative)
r_none, p_none = stats.pearsonr(x, y_none)

print(f"Positive correlation: r = {r_pos:.3f}, p-value = {p_pos:.6f}")
print(f"Negative correlation: r = {r_neg:.3f}, p-value = {p_neg:.6f}")
print(f"No correlation: r = {r_none:.3f}, p-value = {p_none:.3f}")

# Method 2: NumPy
r_numpy = np.corrcoef(x, y_positive)[0, 1]
print(f"NumPy result: {r_numpy:.3f}")

# Method 3: Pandas
df = pd.DataFrame({'x': x, 'y': y_positive})
r_pandas = df['x'].corr(df['y'])
print(f"Pandas result: {r_pandas:.3f}")
```

### Interpretasi Pearson r

| Value | Interpretation |
| ----- | --------------- |
| **0.9 - 1.0** | Very strong positive |
| **0.7 - 0.9** | Strong positive |
| **0.5 - 0.7** | Moderate positive |
| **0.3 - 0.5** | Weak positive |
| **0.0 - 0.3** | Very weak / negligible |
| **-0.3 - 0.0** | Very weak / negligible |
| **-0.5 - -0.3** | Weak negative |
| **-0.7 - -0.5** | Moderate negative |
| **-0.9 - -0.7** | Strong negative |
| **-1.0 - -0.9** | Very strong negative |

### P-value Interpretation

```python
# P-value indicates statistical significance
r, p = stats.pearsonr(x, y_positive)

if p < 0.05:
    print(f"✓ Correlation is statistically significant (p={p:.6f})")
else:
    print(f"✗ Correlation is NOT statistically significant (p={p:.3f})")

# P-value < 0.05: Ada evidence korelasi meaningful, bukan random chance
```

### Visualisasi 3 Cases

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Positive correlation
axes[0].scatter(x, y_positive, alpha=0.6)
axes[0].set_title(f'Positive Correlation\nr = {r_pos:.3f}')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(alpha=0.3)

# Negative correlation
axes[1].scatter(x, y_negative, alpha=0.6)
axes[1].set_title(f'Negative Correlation\nr = {r_neg:.3f}')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].grid(alpha=0.3)

# No correlation
axes[2].scatter(x, y_none, alpha=0.6)
axes[2].set_title(f'No Correlation\nr = {r_none:.3f}')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 📈 2. Spearman Rank Correlation

**Spearman rho** mengukur korelasi **monotonic** (tidak perlu linear). Lebih robust terhadap outliers dan tidak assume normal distribution.

Gunakan Spearman ketika:
- Data tidak normal
- Ada outliers
- Data adalah ordinal (ranking)
- Hubungan non-linear tapi monotonic

### Hitung Spearman Correlation

```python
from scipy.stats import spearmanr

# Data dengan outliers
x_outlier = np.concatenate([np.random.randn(95), [10, 10, 10, 10, 10]])
y_outlier = x_outlier * 2 + np.random.randn(100) * 0.5

# Pearson (sensitive to outliers)
r_pearson, p_pearson = stats.pearsonr(x_outlier, y_outlier)

# Spearman (robust to outliers)
rho, p_spearman = spearmanr(x_outlier, y_outlier)

print(f"Pearson r: {r_pearson:.3f}, p-value: {p_pearson:.6f}")
print(f"Spearman rho: {rho:.3f}, p-value: {p_spearman:.6f}")

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x_outlier, y_outlier, alpha=0.6)
axes[0].set_title(f'Pearson r = {r_pearson:.3f}')
axes[0].grid(alpha=0.3)

axes[1].scatter(x_outlier, y_outlier, alpha=0.6)
axes[1].set_title(f'Spearman rho = {rho:.3f}')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 🎯 3. Kendall Tau Correlation

**Kendall tau** adalah non-parametric measure lain yang robust terhadap outliers. Lebih conservative than Spearman.

```python
from scipy.stats import kendalltau

tau, p_kendall = kendalltau(x, y_positive)
print(f"Kendall tau: {tau:.3f}, p-value: {p_kendall:.6f}")
```

### Perbandingan Correlation Methods

```python
print("Correlation Methods Comparison:")
print("-" * 60)

r_pearson, p_pearson = stats.pearsonr(x, y_positive)
rho_spearman, p_spearman = spearmanr(x, y_positive)
tau_kendall, p_kendall = kendalltau(x, y_positive)

methods = ['Pearson', 'Spearman', 'Kendall']
values = [r_pearson, rho_spearman, tau_kendall]
p_values = [p_pearson, p_spearman, p_kendall]

for method, value, p_val in zip(methods, values, p_values):
    print(f"{method:12} | Correlation: {value:.4f} | P-value: {p_val:.6f}")
```

---

## 📊 Correlation Matrix

Correlation matrix menunjukkan korelasi **antara semua pairs variabel** dalam dataset.

### Hitung Correlation Matrix

```python
# Create dataset
np.random.seed(42)
df = pd.DataFrame({
    'Height': np.random.normal(170, 10, 100),
    'Weight': np.random.normal(70, 15, 100),
    'Age': np.random.randint(20, 60, 100),
    'Income': np.random.normal(50000, 20000, 100)
})

# Create correlation matrix
corr_matrix = df.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Add some realistic correlations
df['Weight'] = df['Height'] * 0.5 + np.random.normal(0, 5, 100)
df['Income'] = df['Age'] * 1000 + np.random.normal(0, 10000, 100)

corr_matrix_realistic = df.corr()
print("\nRealistic Correlation Matrix:")
print(corr_matrix_realistic)
```

### Visualisasi: Correlation Heatmap

```python
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random correlation
corr_random = df[['Height', 'Weight', 'Age', 'Income']].corr()
sns.heatmap(corr_random, annot=True, cmap='coolwarm', center=0, 
            vmin=-1, vmax=1, ax=axes[0], square=True)
axes[0].set_title('Correlation Heatmap (Random)')

# Realistic correlation
corr_realistic = df.corr()
sns.heatmap(corr_realistic, annot=True, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, ax=axes[1], square=True)
axes[1].set_title('Correlation Heatmap (Realistic)')

plt.tight_layout()
plt.show()
```

### Pairplot: Visualisasi All Relationships

```python
# Seaborn pairplot menunjukkan scatter plot untuk setiap pair variabel
sns.pairplot(df[['Height', 'Weight', 'Age', 'Income']], 
             diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot - Relationships Between Variables', y=1.00)
plt.show()
```

---

## ⚠️ Korelasi ≠ Kausalitas!

**CRITICAL MISTAKE**: Banyak orang mengasumsikan korelasi tinggi = hubungan sebab-akibat. **WRONG!**

### Contoh Spurious Correlation

```python
# Generate 2 unrelated variables yang happen to correlate
np.random.seed(42)
years = np.arange(2010, 2021)

# Nicolas Cage films per year (made-up data)
cage_films = [1, 2, 2, 3, 1, 3, 2, 1, 2, 3, 1]

# Swimming pool drownings per year (real-ish data)
drownings = [109, 102, 102, 98, 95, 111, 101, 95, 123, 96, 102]

r, p = stats.pearsonr(cage_films, drownings)
print(f"Correlation: r = {r:.3f}, p-value = {p:.3f}")

plt.figure(figsize=(10, 6))
plt.scatter(cage_films, drownings, s=100, alpha=0.6)
plt.xlabel('Nicolas Cage Films')
plt.ylabel('Swimming Pool Drownings')
plt.title(f'Spurious Correlation: r = {r:.3f}')
plt.grid(alpha=0.3)
plt.show()

# Korelasi tinggi (r=0.666) tapi TIDAK ada causal relationship!
```

### Confounding Variable

```python
# Contoh real: Ice cream sales vs drowning deaths
# Korelasi tinggi karena CONFOUNDING VARIABLE = musim panas

# Both meningkat di musim panas:
# - Penjualan ice cream tinggi
# - Banyak orang berenang → drowning tinggi

# Bukan ice cream → drowning, tapi keduanya → musim panas

print("Remember:")
print("Correlation ≠ Causation")
print("Selalu investigate:")
print("  1. Apakah ada confounding variable?")
print("  2. Apakah temporal order masuk akal? (cause before effect)")
print("  3. Apakah ada mechanism yang reasonable?")
```

---

## 📝 Ringkasan

### Kapan Gunakan Masing-masing?

| Correlation | Use When | Assumptions |
| ----------- | -------- | ----------- |
| **Pearson r** | Linear relationship, both continuous | Normal dist, no outliers |
| **Spearman rho** | Monotonic relationship | More robust, no normality required |
| **Kendall tau** | Conservative estimate | Very robust to outliers |

### Quick Decision Tree

```
Data linear & normal?
├─ YES → Pearson r
└─ NO
   ├─ Has outliers?
   │  └─ YES → Spearman or Kendall
   └─ Non-linear but monotonic?
      └─ YES → Spearman
```

---

## 🧪 Statistical Significance

P-value from correlation test:
- **p < 0.05**: Korelasi statistically significant
- **p ≥ 0.05**: Korelasi NOT statistically significant (could be random)

⚠️ **Important**: Statistically significant ≠ large/meaningful correlation!
- Dengan n besar, r=0.1 bisa significant (p<0.05) tapi effect size kecil
- Dengan n kecil, r=0.7 tidak significant (p>0.05) karena uncertainty

---

## ✏️ Latihan

### Latihan 1: Calculate Correlations

```python
# Load dataset
df = pd.read_csv('dataset.csv')

# Calculate all 3 correlation types
numeric_cols = df.select_dtypes(include=['number']).columns

for col1 in numeric_cols:
    for col2 in numeric_cols:
        if col1 < col2:  # Avoid duplicates
            r_pearson, p_pearson = stats.pearsonr(df[col1], df[col2])
            rho_spearman, p_spearman = spearmanr(df[col1], df[col2])
            
            print(f"{col1} vs {col2}:")
            print(f"  Pearson: r={r_pearson:.3f} (p={p_pearson:.4f})")
            print(f"  Spearman: ρ={rho_spearman:.3f} (p={p_spearman:.4f})")
```

### Latihan 2: Heatmap & Pairplot

```python
# Create correlation heatmap
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Create pairplot
sns.pairplot(df, diag_kind='hist')
plt.show()
```

### Latihan 3: Find Spurious Correlations

1. Find 2 variabel dengan korelasi tinggi (r > 0.7)
2. Analyze: Apakah ada causal relationship?
3. Apakah ada confounding variable?
4. Presentasikan findings

---

## 🔗 Referensi

- [SciPy Correlations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
- [Spurious Correlations](http://www.tylervigen.com/spurious-correlations)
