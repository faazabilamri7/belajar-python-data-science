---
title: Ukuran Penyebaran (Dispersion)
description: Variance, Standard Deviation, Range, IQR
sidebar:
  order: 3
---

## 📏 Apa itu Dispersion?

**Dispersion** mengukur **seberapa menyebar data** dari pusat (mean/median). Penting untuk memahami variability dalam data.

Contoh: Dua kelas punya mean nilai sama (75), tetapi:
- Kelas A: [73, 74, 75, 76, 77] - Nilai rapat di sekitar 75 (low dispersion)
- Kelas B: [50, 60, 75, 90, 100] - Nilai tersebar jauh (high dispersion)

---

## 📐 1. Range (Jangkauan)

Range adalah **selisih antara nilai maksimum dan minimum**.

$$\text{Range} = Max - Min$$

### Hitung Range

```python
import numpy as np
import pandas as pd

data = [70, 75, 80, 85, 90, 95, 100]

# Method 1: Manual
range_manual = max(data) - min(data)
print(f"Range: {range_manual}")  # 30

# Method 2: NumPy
range_numpy = np.max(data) - np.min(data)
print(f"Range (numpy): {range_numpy}")  # 30

# Method 3: Pandas
df = pd.Series(data)
range_pandas = df.max() - df.min()
print(f"Range (pandas): {range_pandas}")  # 30
```

### Kelebihan & Kekurangan

✅ **Pros:**
- Mudah dimengerti
- Cepat dihitung

❌ **Cons:**
- **Sangat sensitive terhadap outlier**
- Hanya menggunakan 2 nilai (max & min)
- Tidak mempertimbangkan data di tengah

### Contoh: Range Sensitive to Outliers

```python
data_normal = [70, 75, 80, 85, 90, 95, 100]
data_with_outlier = [70, 75, 80, 85, 90, 95, 1000]

print(f"Normal range: {max(data_normal) - min(data_normal)}")  # 30
print(f"With outlier: {max(data_with_outlier) - min(data_with_outlier)}")  # 930

# 1 outlier mengubah range drastis!
```

---

## 🔍 2. Variance (Varians)

Variance mengukur **rata-rata dari kuadrat jarak setiap nilai ke mean**. Semakin besar variance, semakin spread out datanya.

$$\sigma^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n}$$

### Hitung Variance

```python
data = [70, 75, 80, 85, 90, 95, 100]

# Step-by-step calculation
mean = np.mean(data)
print(f"Mean: {mean}")  # 85.0

# Calculate deviations
deviations = [x - mean for x in data]
print(f"Deviations: {deviations}")
# [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]

# Square deviations
squared_deviations = [d**2 for d in deviations]
print(f"Squared deviations: {squared_deviations}")
# [225.0, 100.0, 25.0, 0.0, 25.0, 100.0, 225.0]

# Average
variance_manual = sum(squared_deviations) / len(data)
print(f"Variance (manual): {variance_manual:.2f}")  # 100.0

# Method 2: NumPy (population variance)
var_pop = np.var(data)
print(f"Variance (population): {var_pop:.2f}")  # 100.0

# Method 3: NumPy (sample variance, use n-1)
var_sample = np.var(data, ddof=1)
print(f"Variance (sample): {var_sample:.2f}")  # 116.67

# Method 4: Pandas
var_pandas = pd.Series(data).var()
print(f"Variance (pandas): {var_pandas:.2f}")  # 116.67 (default ddof=1)
```

### Population vs Sample Variance

- **Population Variance** (σ²): Gunakan ketika data adalah seluruh populasi → Divide by n
- **Sample Variance** (s²): Gunakan ketika data adalah sample dari populasi → Divide by (n-1)

```python
# Contoh: Nilai 100 siswa dari 1000000 siswa di Indonesia
sample_data = [70, 75, 80, 85, 90, 95, 100]

# Population variance (jika ini semua siswa)
var_pop = np.var(sample_data)

# Sample variance (jika ini sample dari populasi lebih besar)
var_sample = np.var(sample_data, ddof=1)

print(f"Population var: {var_pop:.2f}")
print(f"Sample var: {var_sample:.2f}")

# Sample variance lebih besar karena compensation untuk uncertainty
```

### Interpretasi Variance

```python
# Compare two datasets
np.random.seed(42)
data_low_var = np.random.normal(100, 5, 1000)  # std=5
data_high_var = np.random.normal(100, 20, 1000)  # std=20

print(f"Low var dataset - var: {np.var(data_low_var):.2f}")  # ≈ 25
print(f"High var dataset - var: {np.var(data_high_var):.2f}")  # ≈ 400

# High variance datanya lebih spread out
```

---

## 📊 3. Standard Deviation (Simpangan Baku)

Standard Deviation adalah **akar kuadrat dari variance**. Lebih mudah diinterpretasi karena satuan sama dengan data original.

$$\sigma = \sqrt{\text{Variance}} = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n}}$$

### Hitung Standard Deviation

```python
data = [70, 75, 80, 85, 90, 95, 100]

# Method 1: Manual (dari variance)
var = np.var(data, ddof=1)
std_manual = np.sqrt(var)
print(f"Std (manual): {std_manual:.2f}")  # 10.81

# Method 2: NumPy
std_pop = np.std(data)  # population std
std_sample = np.std(data, ddof=1)  # sample std
print(f"Std (population): {std_pop:.2f}")  # 10.0
print(f"Std (sample): {std_sample:.2f}")  # 10.81

# Method 3: Pandas
std_pandas = pd.Series(data).std()
print(f"Std (pandas): {std_pandas:.2f}")  # 10.81
```

### Kenapa Std Dev Lebih Baik dari Variance?

```python
# Contoh: Nilai ujian (skala 0-100)
nilai = [70, 75, 80, 85, 90, 95, 100]

mean = np.mean(nilai)
var = np.var(nilai, ddof=1)
std = np.std(nilai, ddof=1)

print(f"Mean: {mean:.1f}")
print(f"Variance: {var:.1f}")  # Unit: points²
print(f"Std Dev: {std:.1f}")   # Unit: points

# Interpretation:
# "Variance adalah 116.7 points²" - apa itu points²?
# "Std Dev adalah 10.8 points" - mudah dimengerti!
```

### 68-95-99.7 Rule (Empirical Rule)

Untuk data berdistribusi normal:
- **68%** data dalam μ ± 1σ (1 standard deviation)
- **95%** data dalam μ ± 2σ (2 standard deviations)
- **99.7%** data dalam μ ± 3σ (3 standard deviations)

```python
np.random.seed(42)
data_normal = np.random.normal(100, 15, 10000)

mean = np.mean(data_normal)
std = np.std(data_normal)

# Check percentages
within_1std = np.sum((data_normal >= mean - std) & (data_normal <= mean + std)) / len(data_normal) * 100
within_2std = np.sum((data_normal >= mean - 2*std) & (data_normal <= mean + 2*std)) / len(data_normal) * 100
within_3std = np.sum((data_normal >= mean - 3*std) & (data_normal <= mean + 3*std)) / len(data_normal) * 100

print(f"Within ±1σ: {within_1std:.1f}% (expected: 68%)")  # ≈ 68%
print(f"Within ±2σ: {within_2std:.1f}% (expected: 95%)")  # ≈ 95%
print(f"Within ±3σ: {within_3std:.1f}% (expected: 99.7%)")  # ≈ 99.7%
```

### Visualisasi 68-95-99.7 Rule

```python
import matplotlib.pyplot as plt
from scipy import stats

fig, ax = plt.subplots(figsize=(12, 6))

# Plot normal distribution
x = np.linspace(mean - 4*std, mean + 4*std, 1000)
y = stats.norm.pdf(x, mean, std)
ax.plot(x, y, 'b-', linewidth=2, label='Normal Distribution')

# Fill areas
ax.fill_between(x, y, where=(x >= mean - std) & (x <= mean + std), alpha=0.2, color='green', label='±1σ (68%)')
ax.fill_between(x, y, where=(x >= mean - 2*std) & (x <= mean + 2*std), alpha=0.1, color='orange', label='±2σ (95%)')
ax.fill_between(x, y, where=(x >= mean - 3*std) & (x <= mean + 3*std), alpha=0.05, color='red', label='±3σ (99.7%)')

# Add mean line
ax.axvline(mean, color='k', linestyle='--', linewidth=2, alpha=0.5)

ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('68-95-99.7 Rule for Normal Distribution')
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

---

## 📦 4. Interquartile Range (IQR)

IQR adalah **range antara Q1 (25th percentile) dan Q3 (75th percentile)**. IQR lebih robust terhadap outlier karena hanya mengukur middle 50% data.

$$\text{IQR} = Q3 - Q1$$

### Quartiles & Percentiles

```python
data = sorted([70, 75, 80, 85, 90, 95, 100, 105, 110])

# Percentiles
Q1 = np.percentile(data, 25)  # 25th percentile
Q2 = np.percentile(data, 50)  # 50th percentile (median)
Q3 = np.percentile(data, 75)  # 75th percentile
IQR = Q3 - Q1

print(f"Q1 (25th percentile): {Q1}")  # 80.0
print(f"Q2 (50th percentile/Median): {Q2}")  # 90.0
print(f"Q3 (75th percentile): {Q3}")  # 100.0
print(f"IQR: {IQR}")  # 20.0

# Interpretation:
# 25% data di bawah Q1=80
# 50% data di bawah Q2=90
# 75% data di bawah Q3=100
# Middle 50% data berada dalam range [80, 100]
```

### IQR vs Range

```python
data_normal = [70, 75, 80, 85, 90, 95, 100]
data_with_outlier = [70, 75, 80, 85, 90, 95, 1000]

# Range
range_normal = max(data_normal) - min(data_normal)
range_outlier = max(data_with_outlier) - min(data_with_outlier)

# IQR
IQR_normal = np.percentile(data_normal, 75) - np.percentile(data_normal, 25)
IQR_outlier = np.percentile(data_with_outlier, 75) - np.percentile(data_with_outlier, 25)

print(f"Normal - Range: {range_normal}, IQR: {IQR_normal}")  # Range: 30, IQR: 15
print(f"Outlier - Range: {range_outlier}, IQR: {IQR_outlier}")  # Range: 930, IQR: 15

# Range drastis berubah, tapi IQR tetap sama!
# IQR much more robust!
```

### Boxplot: Visualisasi IQR

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Normal data
data1 = [70, 75, 80, 85, 90, 95, 100]
axes[0].boxplot(data1)
axes[0].set_title('Normal Data')
axes[0].set_ylabel('Value')
axes[0].grid(alpha=0.3)

# With outlier
data2 = [70, 75, 80, 85, 90, 95, 1000]
axes[1].boxplot(data2)
axes[1].set_title('With Outlier')
axes[1].set_ylabel('Value')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Box shows Q1-Q3 (IQR)
# Line inside shows Q2 (median)
# Whiskers show 1.5*IQR
# Dots show outliers (> 1.5*IQR from Q1/Q3)
```

---

## 📊 Perbandingan Measures of Dispersion

| Measure | Formula | Pros | Cons | Use When |
| ------- | ------- | ---- | ---- | -------- |
| **Range** | Max - Min | Simple | Sensitive to outliers | Quick overview |
| **Variance** | Σ(x-μ)²/n | Mathematical properties | Hard to interpret | Mathematical calculations |
| **Std Dev** | √Variance | Same unit as data | More for symmetric data | Primary choice usually |
| **IQR** | Q3 - Q1 | Robust to outliers | Less familiar | Has outliers |

---

## 💡 Praktik: Analyze Dispersion

```python
import pandas as pd
import numpy as np

# Create dataset
np.random.seed(42)
df = pd.DataFrame({
    'class_A': np.random.normal(75, 5, 100),  # Low variance
    'class_B': np.random.normal(75, 15, 100),  # High variance
})

# Compare dispersion
for col in df.columns:
    print(f"\n{col}:")
    print(f"  Range: {df[col].max() - df[col].min():.2f}")
    print(f"  Std Dev: {df[col].std():.2f}")
    print(f"  Variance: {df[col].var():.2f}")
    print(f"  IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
```

---

## 📝 Ringkasan

### Key Takeaways

| Measure | What It Measures | Best For |
| ------- | --------------- | -------- |
| **Range** | Full spread | Quick check (avoid if outliers) |
| **Variance** | Average squared distance from mean | Mathematical properties |
| **Std Dev** | Average distance from mean | Primary choice for reporting |
| **IQR** | Middle 50% spread | Has outliers, skewed data |

### Rule of Thumb

```
1. Selalu plot histogram atau boxplot dulu
2. Normal distribution without outliers → Use Std Dev
3. Has outliers → Use IQR
4. Compare groups → Use all three for complete picture
```

---

## ✏️ Latihan

### Latihan 1: Calculate All Measures

Data: `[78, 82, 90, 85, 88, 72, 95, 88, 79, 91]`

1. Hitung: range, variance, std dev, IQR
2. Visualisasi dengan histogram
3. Interpretasi: data ini punya dispersion tinggi atau rendah?

### Latihan 2: Compare Distributions

Generate 2 datasets:
- Dataset A: `np.random.normal(100, 10, 100)` (std=10)
- Dataset B: `np.random.normal(100, 25, 100)` (std=25)

Compare dispersion measures dan buat boxplot comparison.

---

## 🔗 Referensi

- [NumPy Statistics](https://numpy.org/doc/stable/reference/routines.statistics.html)
- [Pandas Describe](https://pandas.pydata.org/docs/reference/frame.html#descriptive-statistics)
