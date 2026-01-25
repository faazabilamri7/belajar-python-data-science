---
title: Outliers Detection & Handling
description: Mendeteksi dan menangani outliers dalam data
sidebar:
  order: 5
---

## 🎯 Apa itu Outliers?

**Outlier** adalah nilai yang sangat berbeda dari mayoritas data. Outlier bisa disebabkan oleh:

- 📊 Error pengukuran atau data entry
- 📊 Legitimate extreme values (legitimate extremes yang nyata)
- 📊 Anomali atau fraud

---

## 🔍 Metode Deteksi Outliers

### 1. Metode IQR (Interquartile Range)

IQR adalah cara paling umum untuk deteksi outliers. Outlier adalah nilai yang berada di luar 1.5 × IQR dari Q1 dan Q3.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

def detect_outliers_iqr(data, column):
    """Deteksi outliers menggunakan IQR method"""
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    print(f"Column: {column}")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Normal range: {lower_bound:.2f} - {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
    
    return outliers, lower_bound, upper_bound

# Usage
outliers, lb, ub = detect_outliers_iqr(df, 'harga')
print(f"\nOutlier values:\n{outliers}")
```

**IQR Method interpretation:**
- Robust terhadap extreme values
- Standard dalam statistics & analytics
- Cocok untuk most distributions

### 2. Metode Z-Score

Z-Score mengukur berapa banyak standard deviation nilai dari mean. Nilai dengan Z-Score > 3 biasanya outlier (99.7% data dalam 3 sigma).

```python
from scipy import stats

def detect_outliers_zscore(data, column, threshold=3):
    """Deteksi outliers menggunakan Z-Score"""
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    threshold_value = threshold
    
    outliers = data[np.abs(stats.zscore(data[column].dropna())) > threshold_value]
    
    print(f"Column: {column}")
    print(f"Threshold Z-Score: {threshold}")
    print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
    
    return outliers

# Usage
# Z-score > 3 (99.7% confidence)
outliers_3 = detect_outliers_zscore(df, 'harga', threshold=3)

# Z-score > 2 (95% confidence, more sensitive)
outliers_2 = detect_outliers_zscore(df, 'harga', threshold=2)
```

**Z-Score method:**
- Assume normal distribution
- Sensitive dengan threshold choice
- Cocok untuk normally distributed data

### 3. Metode MAD (Median Absolute Deviation)

Lebih robust daripada Z-Score karena menggunakan median instead of mean.

```python
def detect_outliers_mad(data, column, threshold=2.5):
    """Deteksi outliers menggunakan MAD method"""
    
    median = data[column].median()
    mad = np.median(np.abs(data[column] - median))
    
    # Modified Z-scores
    modified_z_scores = 0.6745 * (data[column] - median) / mad
    
    outliers = data[np.abs(modified_z_scores) > threshold]
    
    print(f"Column: {column}")
    print(f"Median: {median:.2f}, MAD: {mad:.2f}")
    print(f"Number of outliers: {len(outliers)}")
    
    return outliers

# Usage
outliers_mad = detect_outliers_mad(df, 'harga')
```

---

## 📊 Visualisasi Outliers

### Boxplot

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Single boxplot
plt.figure(figsize=(8, 6))
df['harga'].plot(kind='box')
plt.title('Harga - Boxplot (Outliers di luar whiskers)')
plt.ylabel('Harga')
plt.show()

# Multiple boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(df['harga'].dropna())
axes[0].set_title('Harga')

axes[1].boxplot(df['nilai'].dropna())
axes[1].set_title('Nilai')

plt.tight_layout()
plt.show()

# Seaborn boxplot dengan hue
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='kategori', y='harga')
plt.title('Harga by Kategori')
plt.show()
```

### Scatter Plot + Outlier Highlighting

```python
# Scatter plot dengan outlier highlight
Q1 = df['harga'].quantile(0.25)
Q3 = df['harga'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = (df['harga'] < lower_bound) | (df['harga'] > upper_bound)

plt.figure(figsize=(12, 6))
plt.scatter(df[~outlier_mask]['quantity'], df[~outlier_mask]['harga'], 
           alpha=0.6, label='Normal', s=50)
plt.scatter(df[outlier_mask]['quantity'], df[outlier_mask]['harga'], 
           alpha=0.8, color='red', label='Outliers', s=100)
plt.xlabel('Quantity')
plt.ylabel('Harga')
plt.legend()
plt.title('Outliers Detection')
plt.show()
```

### Distribution Plot

```python
# Histogram dengan normal distribution overlay
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Dengan outliers
axes[0].hist(df['harga'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(lower_bound, color='red', linestyle='--', label=f'Lower: {lower_bound:.0f}')
axes[0].axvline(upper_bound, color='red', linestyle='--', label=f'Upper: {upper_bound:.0f}')
axes[0].set_title('Distribution (dengan outliers)')
axes[0].legend()

# Tanpa outliers
axes[1].hist(df[~outlier_mask]['harga'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[1].set_title('Distribution (tanpa outliers)')

plt.tight_layout()
plt.show()
```

---

## 🔧 Strategi Menangani Outliers

### 1. Remove (Penghapusan)

Hapus rows dengan outliers - paling sederhana tapi hilang data.

```python
# Remove menggunakan IQR method
Q1 = df['harga'].quantile(0.25)
Q3 = df['harga'].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[
    (df['harga'] >= Q1 - 1.5 * IQR) &
    (df['harga'] <= Q3 + 1.5 * IQR)
]

print(f"Rows sebelum: {len(df)}, sesudah: {len(df_clean)}")

# Remove menggunakan Z-score
df_clean = df[np.abs(stats.zscore(df['harga'])) < 3]
```

**Kapan gunakan:**
- ✅ Outliers adalah errors (data entry mistakes)
- ✅ Outliers sedikit (< 5%)
- ✅ Outliers tidak informatif

### 2. Capping / Winsorization

Batas outliers ke nilai tertentu (biasanya percentile).

```python
# Capping ke 5th dan 95th percentile
lower = df['harga'].quantile(0.05)
upper = df['harga'].quantile(0.95)

df['harga_capped'] = df['harga'].clip(lower, upper)

# Verify
print(f"Original min-max: {df['harga'].min():.2f} - {df['harga'].max():.2f}")
print(f"Capped min-max: {df['harga_capped'].min():.2f} - {df['harga_capped'].max():.2f}")

# Capping dengan IQR bounds
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df['harga_capped'] = df['harga'].clip(lower, upper)
```

**Kapan gunakan:**
- ✅ Outliers adalah legitimate extremes
- ✅ Ingin preserve informasi tapi limit impact
- ✅ For modeling dengan sensitive algorithms (KNN, clustering)

### 3. Transformation (Log, Square Root)

Transform data untuk reduce outlier impact sambil preserve relationships.

```python
# Log transformation (untuk right-skewed data)
df['harga_log'] = np.log1p(df['harga'])  # log(1 + x) untuk handle 0 values

# Square root transformation (milder daripada log)
df['harga_sqrt'] = np.sqrt(df['harga'])

# Box-Cox transformation (optimal transformation)
from scipy.stats import boxcox
df['harga_boxcox'], lambda_param = boxcox(df['harga'] + 1)

# Visualisasi perbandingan
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].hist(df['harga'], bins=30, edgecolor='black')
axes[0,0].set_title('Original')

axes[0,1].hist(df['harga_log'], bins=30, edgecolor='black')
axes[0,1].set_title('Log Transform')

axes[1,0].hist(df['harga_sqrt'], bins=30, edgecolor='black')
axes[1,0].set_title('Square Root Transform')

axes[1,1].hist(df['harga_boxcox'], bins=30, edgecolor='black')
axes[1,1].set_title('Box-Cox Transform')

plt.tight_layout()
plt.show()
```

**Kapan gunakan:**
- ✅ Outliers adalah legitimate extremes
- ✅ Model sensitive dengan skewed distribution
- ✅ Want to preserve all data points

### 4. Separate Analysis

Treat outliers sebagai separate group dan analyze terpisah.

```python
# Buat flag untuk outliers
outlier_mask = (df['harga'] < lower_bound) | (df['harga'] > upper_bound)
df['is_outlier'] = outlier_mask.astype(int)

# Analyze normal dan outliers terpisah
print("=== NORMAL DATA ===")
print(df[~outlier_mask]['harga'].describe())

print("\n=== OUTLIERS ===")
print(df[outlier_mask]['harga'].describe())

# Group analysis
print("\n=== PER KATEGORI ===")
print(df.groupby(['kategori', 'is_outlier'])['harga'].describe())
```

---

## 📋 Outlier Handling Decision Tree

```
Ada outliers?
├─ Yes, data entry errors
│  └─ Remove (drop rows)
├─ Yes, legitimate extremes
│  ├─ Penting untuk model?
│  │  ├─ Yes
│  │  │  └─ Capping atau Transform
│  │  └─ No
│  │     └─ Remove atau Capping
└─ No
   └─ Continue analysis
```

---

## ✏️ Praktik: Outlier Detection & Handling

```python
import pandas as pd
import numpy as np
from scipy import stats

# Create sample data dengan outliers
np.random.seed(42)
data = {
    'harga': np.concatenate([
        np.random.normal(100, 20, 95),  # 95 normal values
        [250, 280, 300, 350, 400]       # 5 outliers
    ]),
    'kategori': np.random.choice(['A', 'B', 'C'], 100)
}

df = pd.DataFrame(data)

print("=== ORIGINAL DATA ===")
print(df['harga'].describe())

# Deteksi outliers
Q1 = df['harga'].quantile(0.25)
Q3 = df['harga'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['harga'] < lower) | (df['harga'] > upper)]
print(f"\nOutliers: {len(outliers)} detected")

# Strategy 1: Remove
df_remove = df[(df['harga'] >= lower) & (df['harga'] <= upper)]
print(f"\nStrategy 1 - Remove: {len(df)} → {len(df_remove)}")

# Strategy 2: Capping
df_cap = df.copy()
df_cap['harga'] = df_cap['harga'].clip(lower, upper)
print(f"Strategy 2 - Capping: mean before={df['harga'].mean():.2f}, after={df_cap['harga'].mean():.2f}")

# Strategy 3: Transform
df_log = df.copy()
df_log['harga'] = np.log1p(df_log['harga'])
print(f"Strategy 3 - Log Transform: skewness before={df['harga'].skew():.2f}, after={df_log['harga'].skew():.2f}")
```

---

## 📝 Ringkasan Halaman Ini

### Outlier Detection & Handling

| Method | Pros | Cons |
| ------ | ---- | ---- |
| IQR | Robust, simple | Needs assumption |
| Z-Score | Statistical | Assume normality |
| Visual (Boxplot) | Intuitive | Subjective |
| Remove | Simple | Lose data |
| Capping | Preserve data | Artificial bounds |
| Transform | Reduce impact | Change meaning |

---

## ✏️ Latihan

### Latihan 1: Detect Outliers

```python
df = sns.load_dataset('tips')

# 1. IQR method
Q1 = df['total_bill'].quantile(0.25)
Q3 = df['total_bill'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['total_bill'] < lower) | (df['total_bill'] > upper)]
print(f"Outliers: {len(outliers)}")

# 2. Visualisasi
plt.boxplot(df['total_bill'])
plt.show()
```

### Latihan 2: Handle Outliers

```python
# Remove
df_remove = df[(df['total_bill'] >= lower) & (df['total_bill'] <= upper)]

# Cap
df_cap = df.copy()
df_cap['total_bill'] = df_cap['total_bill'].clip(lower, upper)

# Transform
df_log = df.copy()
df_log['total_bill'] = np.log1p(df_log['total_bill'])

print(f"Original: {len(df)}")
print(f"Remove: {len(df_remove)}")
print(f"Cap: {len(df_cap)}")
```

---

## 🔗 Referensi

- [Scipy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Outlier Detection Methods](https://en.wikipedia.org/wiki/Outlier#Detection)
