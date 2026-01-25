---
title: Distribusi Data
description: Normal Distribution, Skewness, Kurtosis
sidebar:
  order: 4
---

## 📊 Apa itu Distribusi?

**Distribusi** mendeskripsikan **bagaimana data tersebar** - frekuensi setiap nilai atau range nilai dalam dataset. Memahami distribusi crucial untuk memilih model dan melakukan statistical tests.

---

## 🔔 1. Normal Distribution (Gaussian Distribution)

Normal Distribution adalah distribusi paling penting dalam statistik. Banyak fenomena alam mengikuti distribusi normal:
- Tinggi badan manusia
- IQ
- Measurement errors
- Banyak biological measurements

### Karakteristik Normal Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate normal distribution
np.random.seed(42)
data_normal = np.random.normal(loc=100, scale=15, size=10000)

# Plot
plt.figure(figsize=(10, 6))
plt.hist(data_normal, bins=50, density=True, alpha=0.7, edgecolor='black', label='Data')

# Plot theoretical normal curve
x = np.linspace(data_normal.min(), data_normal.max(), 100)
plt.plot(x, stats.norm.pdf(x, 100, 15), 'r-', linewidth=2, label='Normal PDF')

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Normal Distribution (μ=100, σ=15)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Properties

✅ **Symmetric** - Ekor kiri = ekor kanan
✅ **Bell-shaped** - Puncak di tengah
✅ **Unimodal** - 1 peak (mode = mean = median)
✅ **68-95-99.7 rule** - Persentase fixed di setiap σ

### Standar Normal Distribution (Z-Distribution)

Special case: mean=0, std=1

```python
# Standard normal distribution
data_standard = np.random.standard_normal(10000)

plt.figure(figsize=(10, 6))
plt.hist(data_standard, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.axvline(0, color='r', linestyle='--', linewidth=2, label='Mean = 0')
plt.axvline(-1, color='g', linestyle='--', alpha=0.5, label='±1σ')
plt.axvline(1, color='g', linestyle='--', alpha=0.5)
plt.title('Standard Normal Distribution (Z-Distribution)')
plt.xlabel('Z-score')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## 🔀 2. Skewness (Kemiringan)

**Skewness** mengukur **asimetri distribusi**. Menunjukkan apakah data skewed ke kiri atau kanan.

$$\text{Skewness} = \frac{\sum (x_i - \bar{x})^3 / n}{\sigma^3}$$

### Interpretasi Skewness

```python
from scipy.stats import skew

# Right-skewed (positively skewed)
data_right = np.random.exponential(2, 1000)
skew_right = skew(data_right)

# Left-skewed (negatively skewed)
data_left = -np.random.exponential(2, 1000)
skew_left = skew(data_left)

# Symmetric (normal)
data_normal = np.random.normal(0, 1, 1000)
skew_normal = skew(data_normal)

print(f"Right-skewed: {skew_right:.3f}")  # > 0
print(f"Left-skewed: {skew_left:.3f}")   # < 0
print(f"Symmetric: {skew_normal:.3f}")   # ≈ 0
```

### Visualisasi Skewness

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Right-skewed
data_right = np.random.exponential(2, 1000)
axes[0].hist(data_right, bins=40, edgecolor='black', alpha=0.7)
axes[0].axvline(np.mean(data_right), color='r', linestyle='--', label=f'Mean: {np.mean(data_right):.2f}')
axes[0].axvline(np.median(data_right), color='g', linestyle='--', label=f'Median: {np.median(data_right):.2f}')
axes[0].set_title(f'Right-Skewed\n(Skewness: {skew(data_right):.2f})')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Symmetric
data_normal = np.random.normal(0, 1, 1000)
axes[1].hist(data_normal, bins=40, edgecolor='black', alpha=0.7)
axes[1].axvline(np.mean(data_normal), color='r', linestyle='--', label=f'Mean: {np.mean(data_normal):.2f}')
axes[1].axvline(np.median(data_normal), color='g', linestyle='--', label=f'Median: {np.median(data_normal):.2f}')
axes[1].set_title(f'Symmetric (Normal)\n(Skewness: {skew(data_normal):.2f})')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Left-skewed
data_left = -np.random.exponential(2, 1000)
axes[2].hist(data_left, bins=40, edgecolor='black', alpha=0.7)
axes[2].axvline(np.mean(data_left), color='r', linestyle='--', label=f'Mean: {np.mean(data_left):.2f}')
axes[2].axvline(np.median(data_left), color='g', linestyle='--', label=f'Median: {np.median(data_left):.2f}')
axes[2].set_title(f'Left-Skewed\n(Skewness: {skew(data_left):.2f})')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Hubungan Skewness dengan Mean & Median

| Condition | Mean vs Median | Skewness |
| --------- | ------------- | --------- |
| **Symmetric** | Mean ≈ Median | ≈ 0 |
| **Right-skewed** | Mean > Median | > 0 (positive tail) |
| **Left-skewed** | Mean < Median | < 0 (negative tail) |

### Interpretasi Values

- **-1 < Skewness < 1**: Approximately symmetric
- **1 < \|Skewness\| < 2**: Moderately skewed
- **\|Skewness\| > 2**: Highly skewed

---

## ⛰️ 3. Kurtosis

**Kurtosis** mengukur **"keruncingan" atau "ekor tebalnya" distribusi**. Membandingkan dengan normal distribution.

$$\text{Kurtosis} = \frac{\sum (x_i - \bar{x})^4 / n}{\sigma^4} - 3$$

(Excess kurtosis = kurtosis - 3, supaya normal distribution = 0)

### Jenis-jenis Kurtosis

```python
from scipy.stats import kurtosis

# Generate different distributions
np.random.seed(42)

# Mesokurtic (like normal distribution)
data_meso = np.random.normal(0, 1, 10000)

# Leptokurtic (more peaked, fatter tails)
data_lepto = np.random.standard_t(3, size=10000)  # t-distribution

# Platykurtic (more flat, thinner tails)
data_platy = np.random.uniform(-3, 3, 10000)

kurt_meso = kurtosis(data_meso)
kurt_lepto = kurtosis(data_lepto)
kurt_platy = kurtosis(data_platy)

print(f"Mesokurtic (normal): {kurt_meso:.3f}")    # ≈ 0
print(f"Leptokurtic (peaked): {kurt_lepto:.3f}")  # > 0
print(f"Platykurtic (flat): {kurt_platy:.3f}")    # < 0
```

### Visualisasi Kurtosis

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Mesokurtic (Normal)
data_meso = np.random.normal(0, 1, 10000)
axes[0].hist(data_meso, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0].set_title(f'Mesokurtic (Normal)\nKurtosis: {kurtosis(data_meso):.2f}')
axes[0].grid(alpha=0.3)

# Leptokurtic (Peaked)
data_lepto = np.random.standard_t(3, size=10000)
axes[1].hist(data_lepto, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[1].set_xlim(-4, 4)
axes[1].set_title(f'Leptokurtic (Peaked)\nKurtosis: {kurtosis(data_lepto):.2f}')
axes[1].grid(alpha=0.3)

# Platykurtic (Flat)
data_platy = np.random.uniform(-3, 3, 10000)
axes[2].hist(data_platy, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[2].set_title(f'Platykurtic (Flat)\nKurtosis: {kurtosis(data_platy):.2f}')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Interpretasi Kurtosis

| Value | Type | Interpretation |
| ----- | ---- | --------------- |
| **Kurtosis ≈ 0** | Mesokurtic | Similar to normal distribution |
| **Kurtosis > 0** | Leptokurtic | More peaked, fatter tails (outliers) |
| **Kurtosis < 0** | Platykurtic | More flat, thinner tails |

---

## 🔍 Mendeteksi Normalitas

### Visual Methods

#### 1. Histogram
```python
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram Check for Normality')
plt.show()
# Look for bell-shaped, symmetric curve
```

#### 2. Q-Q Plot
```python
from scipy import stats

fig, ax = plt.subplots(figsize=(8, 6))
stats.probplot(data, dist="norm", plot=ax)
ax.set_title('Q-Q Plot')
plt.show()
# If points follow diagonal line → normal
```

#### 3. Kernel Density Estimate
```python
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data, kde=True, bins=30)
plt.title('Distribution with KDE')
plt.show()
```

### Statistical Tests

#### Shapiro-Wilk Test
```python
from scipy.stats import shapiro

# Null hypothesis: data is normally distributed
statistic, p_value = shapiro(data)

print(f"Shapiro-Wilk Test:")
print(f"  Statistic: {statistic:.4f}")
print(f"  P-value: {p_value:.4f}")

if p_value > 0.05:
    print("  ✓ Data appears normally distributed (fail to reject null)")
else:
    print("  ✗ Data does NOT appear normally distributed (reject null)")
```

#### Kolmogorov-Smirnov Test
```python
from scipy.stats import kstest

# Test against standard normal
statistic, p_value = kstest(data, 'norm')

print(f"Kolmogorov-Smirnov Test:")
print(f"  Statistic: {statistic:.4f}")
print(f"  P-value: {p_value:.4f}")
```

#### Anderson-Darling Test
```python
from scipy.stats import anderson

result = anderson(data, dist='norm')

print(f"Anderson-Darling Test:")
print(f"  Statistic: {result.statistic:.4f}")
print(f"  Critical values: {result.critical_values}")
print(f"  Significance levels: {result.significance_level}%")
```

---

## 📐 Z-Score Recap

Z-score menggunakan normal distribution untuk standardize data.

$$z = \frac{x - \mu}{\sigma}$$

```python
from scipy import stats

data = [70, 75, 80, 85, 90, 95, 100]

# Calculate z-scores
z_scores = stats.zscore(data)
print(f"Z-scores: {z_scores}")

# Example: value 95
value = 95
mean = np.mean(data)
std = np.std(data, ddof=1)
z = (value - mean) / std
print(f"\nZ-score for {value}: {z:.2f}")

# Interpretation:
# z = 1.5 means: 1.5 standard deviations above mean
# z = -2.0 means: 2 standard deviations below mean
# z > 3 or z < -3: likely outlier (99.7% rule)
```

---

## 📝 Ringkasan

### Distribution Types & Properties

| Distribution | Shape | Skewness | Kurtosis | When Observed |
| ------------ | ----- | -------- | -------- | ------------- |
| **Normal** | Bell, symmetric | ≈ 0 | ≈ 0 | Many natural phenomena |
| **Right-skewed** | Tail to right | > 0 | Often > 0 | Income, salary data |
| **Left-skewed** | Tail to left | < 0 | Often > 0 | Exam scores (ceiling effect) |
| **Uniform** | Flat | ≈ 0 | < 0 | Random numbers |
| **Bimodal** | 2 peaks | Varies | Varies | Mixed populations |

---

## ✏️ Latihan

### Latihan 1: Identify Distribution Type

Generate 3 datasets:
1. Normal distribution
2. Right-skewed distribution
3. Left-skewed distribution

Untuk setiap:
- Plot histogram
- Calculate skewness & kurtosis
- Identify distribution type

### Latihan 2: Test Normality

```python
# Load data
df = pd.read_csv('dataset.csv')

# For each numeric column
for col in df.select_dtypes(include=['number']).columns:
    # Visual test
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(df[col], bins=30, edgecolor='black')
    axes[0].set_title(f'{col} - Histogram')
    
    stats.probplot(df[col], dist="norm", plot=axes[1])
    axes[1].set_title(f'{col} - Q-Q Plot')
    
    plt.show()
    
    # Statistical test
    stat, p = shapiro(df[col])
    print(f"{col}: p-value = {p:.4f} ({'Normal' if p > 0.05 else 'Not Normal'})")
```

### Latihan 3: Transformation

```python
# If data is right-skewed, try transformations
data_right_skewed = np.random.exponential(2, 1000)

# Try different transformations
log_data = np.log(data_right_skewed)
sqrt_data = np.sqrt(data_right_skewed)
boxcox_data, lambda_param = stats.boxcox(data_right_skewed)

# Compare before and after
for transformed, name in [(data_right_skewed, 'Original'),
                          (log_data, 'Log'),
                          (sqrt_data, 'Sqrt'),
                          (boxcox_data, 'Box-Cox')]:
    skewness = skew(transformed)
    print(f"{name}: Skewness = {skewness:.3f}")
```

---

## 🔗 Referensi

- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Normalcy Tests](https://en.wikipedia.org/wiki/Normality_test)
