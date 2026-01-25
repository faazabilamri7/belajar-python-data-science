---
title: Ukuran Pemusatan (Central Tendency)
description: Mean, Median, Mode dan cara menggunakannya
sidebar:
  order: 2
---

## 📊 Apa itu Central Tendency?

**Central Tendency** mengukur "pusat" dari suatu dataset - nilai yang merepresentasikan seluruh data. Ada 3 ukuran utama: Mean, Median, dan Mode.

Mari kita gunakan contoh data nilai siswa:
```
85, 90, 78, 92, 88, 95, 70, 85, 91, 89
```

---

## 📈 1. Mean (Rata-rata)

Mean adalah **sum dari semua nilai dibagi jumlah data**. Mean paling umum digunakan untuk meringkas data.

$$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$$

### Hitung Mean

```python
import numpy as np
import pandas as pd

data = [85, 90, 78, 92, 88, 95, 70, 85, 91, 89]

# Method 1: Manual
mean_manual = sum(data) / len(data)
print(f"Mean (manual): {mean_manual:.2f}")  # 86.30

# Method 2: NumPy
mean_numpy = np.mean(data)
print(f"Mean (numpy): {mean_numpy:.2f}")  # 86.30

# Method 3: Pandas
mean_pandas = pd.Series(data).mean()
print(f"Mean (pandas): {mean_pandas:.2f}")  # 86.30
```

### Kapan Menggunakan Mean?

✅ **Gunakan Mean ketika:**
- Data berdistribusi normal (simetris)
- Tidak ada outlier ekstrem
- Ingin "average" value yang representative

❌ **Jangan gunakan Mean ketika:**
- Ada outlier ekstrem (akan pulled up/down)
- Data sangat skewed (tidak simetris)
- Data sangat kecil (n < 10, gunakan pertimbangan lain juga)

### Contoh: Outlier Mempengaruhi Mean

```python
data_normal = [85, 90, 78, 92, 88]
data_with_outlier = [85, 90, 78, 92, 88, 1000]  # Ada outlier 1000

print(f"Mean normal: {np.mean(data_normal):.2f}")  # 86.60
print(f"Mean with outlier: {np.mean(data_with_outlier):.2f}")  # 238.83

# Mean berubah drastis karena 1 outlier!
# Ini menunjukkan mean sensitive terhadap extreme values
```

---

## 🎯 2. Median (Nilai Tengah)

Median adalah **nilai yang membagi dataset menjadi dua bagian sama besar**. 50% data di bawah median, 50% di atas. Median lebih **robust** terhadap outlier dibanding mean.

### Cara Menghitung Median

```python
data = [85, 90, 78, 92, 88, 95, 70, 85, 91, 89]

# Urutkan terlebih dahulu
data_sorted = sorted(data)
print(f"Sorted: {data_sorted}")
# [70, 78, 85, 85, 88, 89, 90, 91, 92, 95]

# Jika n ganjil: ambil nilai tengah
# Jika n genap: rata-rata dari 2 nilai tengah

# Method 1: Manual
n = len(data)
if n % 2 == 1:
    median_manual = data_sorted[n // 2]
else:
    median_manual = (data_sorted[n // 2 - 1] + data_sorted[n // 2]) / 2

print(f"Median (manual): {median_manual:.2f}")  # 88.50

# Method 2: NumPy
median_numpy = np.median(data)
print(f"Median (numpy): {median_numpy:.2f}")  # 88.50

# Method 3: Pandas
median_pandas = pd.Series(data).median()
print(f"Median (pandas): {median_pandas:.2f}")  # 88.50
```

### Median Robust Terhadap Outlier

```python
data_normal = [85, 90, 78, 92, 88, 95, 70, 85, 91, 89]
data_with_outlier = [85, 90, 78, 92, 88, 95, 70, 85, 91, 1000]

mean_normal = np.mean(data_normal)
median_normal = np.median(data_normal)

mean_outlier = np.mean(data_with_outlier)
median_outlier = np.median(data_with_outlier)

print(f"Normal - Mean: {mean_normal:.2f}, Median: {median_normal:.2f}")
# Normal - Mean: 86.30, Median: 88.50

print(f"Outlier - Mean: {mean_outlier:.2f}, Median: {median_outlier:.2f}")
# Outlier - Mean: 169.30, Median: 89.00

# Mean berubah drastis (86.30 → 169.30)
# Median hanya sedikit berubah (88.50 → 89.00)
# Ini kenapa median lebih robust!
```

### Kapan Menggunakan Median?

✅ **Gunakan Median ketika:**
- Data memiliki outlier ekstrem
- Data skewed (tidak simetris)
- Data adalah ranking atau ordinal
- Dataset sangat kecil

❌ **Jangan gunakan Median ketika:**
- Data berdistribusi normal dan tanpa outlier
- Butuh statistik lebih lanjut (mean lebih fleksibel untuk calculations)

---

## 🏆 3. Mode (Nilai Paling Sering Muncul)

Mode adalah **nilai yang paling sering muncul** dalam dataset. Mode sangat berguna untuk data kategorikal.

### Hitung Mode

```python
from scipy import stats

# Data numeric
data = [1, 2, 2, 3, 3, 3, 4, 4, 5]

# Method 1: SciPy
mode_result = stats.mode(data, keepdims=True)
print(f"Mode: {mode_result.mode[0]}")  # 3

# Method 2: Pandas
mode_pandas = pd.Series(data).mode()[0]
print(f"Mode (pandas): {mode_pandas}")  # 3

# Data kategorikal
kategori = ['A', 'B', 'A', 'C', 'A', 'B', 'B']

# Find mode
mode_kat = pd.Series(kategori).mode()[0]
print(f"Mode kategori: {mode_kat}")  # B (muncul 3x)

# Lihat frequency
print(pd.Series(kategori).value_counts())
# B    3
# A    3
# C    1
```

### Multi-Modal Data

```python
# Data dengan multiple modes (bimodal)
data_bimodal = [1, 1, 1, 2, 2, 2, 5]  # Dua mode: 1 dan 2

mode_result = stats.mode(data_bimodal, keepdims=True)
print(f"Mode: {mode_result.mode[0]}")  # 1 (hanya return 1 mode)

# Check semua frequencies
print(pd.Series(data_bimodal).value_counts())
# 1    3
# 2    3
# 5    1
```

### Kapan Menggunakan Mode?

✅ **Gunakan Mode ketika:**
- Data kategorikal (warna, kategori, label)
- Mencari nilai yang paling umum/frequent
- Data adalah nominal (tidak ada order)

❌ **Jangan gunakan Mode ketika:**
- Data numeric dan continuous
- Data memiliki multiple modes (ambiguous)

---

## 📊 Perbandingan Mean, Median, Mode

### Karakteristik Masing-masing

| Karakteristik | Mean | Median | Mode |
| ------------- | ---- | ------ | ---- |
| **Definisi** | Rata-rata semua nilai | Nilai tengah | Nilai paling sering |
| **Sensitivity ke Outlier** | Tinggi | Rendah | Tidak affected |
| **Data Type** | Numeric | Numeric/Ordinal | Numeric/Categorical |
| **Formula** | Σx/n | Tengah dari sorted | Max frequency |
| **Robustness** | Low | High | Very High |
| **Use Case** | Normal distribution | Skewed data | Categorical data |

### Visualisasi Perbandingan

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create different distributions
np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Normal Distribution
ax = axes[0, 0]
data_normal = np.random.normal(100, 15, 1000)
ax.hist(data_normal, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(data_normal), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data_normal):.1f}')
ax.axvline(np.median(data_normal), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(data_normal):.1f}')
ax.set_title('Normal Distribution\n(Mean ≈ Median)')
ax.legend()
ax.grid(alpha=0.3)

# 2. Right-Skewed Distribution
ax = axes[0, 1]
data_right = np.random.exponential(2, 1000)
ax.hist(data_right, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(data_right), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data_right):.2f}')
ax.axvline(np.median(data_right), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(data_right):.2f}')
ax.set_title('Right-Skewed Distribution\n(Mean > Median)')
ax.legend()
ax.grid(alpha=0.3)

# 3. Left-Skewed Distribution
ax = axes[1, 0]
data_left = -np.random.exponential(2, 1000)
ax.hist(data_left, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(data_left), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data_left):.2f}')
ax.axvline(np.median(data_left), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(data_left):.2f}')
ax.set_title('Left-Skewed Distribution\n(Mean < Median)')
ax.legend()
ax.grid(alpha=0.3)

# 4. With Outliers
ax = axes[1, 1]
data_with_outliers = np.concatenate([np.random.normal(100, 15, 950), np.array([500, 520, 530, 540])])
ax.hist(data_with_outliers, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(data_with_outliers), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data_with_outliers):.1f}')
ax.axvline(np.median(data_with_outliers), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(data_with_outliers):.1f}')
ax.set_title('With Outliers\n(Median more robust)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 💡 Rule of Thumb: Kapan Gunakan Apa?

```
1. Hitung mean, median, dan lihat hasilnya
   ├─ Jika Mean ≈ Median → Gunakan Mean (normal distribution)
   ├─ Jika |Mean - Median| > 5% → Ada skewness atau outliers
   │  └─ Gunakan Median untuk summary
   └─ Selalu visualisasi histogram untuk verify!

2. Untuk data kategorikal → Gunakan Mode

3. Untuk presentasi → Gunakan Mean (paling familiar)
   Untuk analisis → Gunakan Median (lebih robust)
```

---

## 📝 Ringkasan

### Key Takeaways

| Ukuran | Definisi | Best For | Pros | Cons |
| ------ | -------- | -------- | ---- | ---- |
| **Mean** | Σx / n | Normal data | Easy to compute, familiar | Sensitive to outliers |
| **Median** | Middle value | Skewed data | Robust to outliers | Less familiar, harder to compute |
| **Mode** | Most frequent | Categorical | Robust, works for any type | Can be ambiguous |

---

## ✏️ Latihan

### Latihan 1: Compute Central Tendency

Diberikan data nilai mahasiswa: `[78, 82, 90, 85, 88, 72, 95, 88, 79, 91]`

1. Hitung mean, median, dan mode
2. Manakah yang paling representative untuk dataset ini?
3. Apa yang akan terjadi jika ada outlier nilai 5?

```python
# Your code here
```

### Latihan 2: Compare Distributions

Generate 3 datasets berbeda:
1. Normal distribution
2. Right-skewed distribution
3. Data dengan outliers

Untuk setiap dataset:
- Hitung mean, median, mode
- Plot histogram dengan mean & median lines
- Jelaskan differences

### Latihan 3: Real Data Analysis

```python
# Load real dataset
df = pd.read_csv('your_dataset.csv')

# Analyze central tendency per group
for col in df.select_dtypes(include=['number']).columns:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Mode: {df[col].mode()[0]}")
    
    # Group analysis
    if 'group' in df.columns:
        print(f"\n  By group:")
        print(df.groupby('group')[col].agg(['mean', 'median']))
```

---

## 🔗 Referensi

- [NumPy Mean, Median, Mode](https://numpy.org/doc/stable/reference/statistics.html)
- [Pandas Describe](https://pandas.pydata.org/docs/reference/frame.html#descriptive-statistics)
