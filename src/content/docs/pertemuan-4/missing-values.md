---
title: Missing Values Handling
description: Strategi mengidentifikasi dan menangani missing values
sidebar:
  order: 4
---

## 📭 Missing Values Problem

Missing values adalah salah satu masalah paling umum dalam dataset real-world. Kita harus mengidentifikasi ada missing values di mana, berapa banyak, dan menggunakan strategi yang tepat untuk menanganinya.

### Penyebab Missing Values

- 🔴 Data tidak dikumpulkan (device rusak, tidak ada input)
- 🔴 Data tidak tersedia (private information)
- 🔴 Data entry errors (typo → dihapus)
- 🔴 Data processing errors

---

## 🔍 Identifikasi Missing Values

### Deteksi Awal

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Cek missing values
print(df.isnull().sum())
print(df.isnull().head())

# Ada missing?
print(df.isnull().any())

# Total missing
print(f"Total missing: {df.isnull().sum().sum()}")
```

### Analisis Missing

```python
# Persentase missing per kolom
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

# Summary table
missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False))

# Baris dengan missing values
print(df[df.isnull().any(axis=1)].head())

# Jumlah missing per baris
rows_with_missing = df.isnull().sum(axis=1)
print(rows_with_missing[rows_with_missing > 0].head())
```

### Visualisasi Missing Values

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Bar chart missing %
plt.figure(figsize=(10, 6))
missing_pct[missing_pct > 0].sort_values().plot(kind='barh', color='coral')
plt.xlabel('Percentage Missing (%)')
plt.title('Missing Values by Column')
plt.show()
```

---

## 🧹 Strategi Menangani Missing Values

Ada berbagai strategi tergantung pada konteks dan persentase missing data.

### 1. Drop Rows (Penghapusan Baris)

Paling sederhana - hapus baris yang punya missing values.

**Kapan gunakan:**
- ✅ Missing sedikit (< 5%)
- ✅ Baris yang hilang tidak penting
- ✅ Random missing (bukan pattern)

```python
# Hapus baris dengan ANY missing values
df_clean = df.dropna()
print(f"Rows sebelum: {len(df)}, sesudah: {len(df_clean)}")

# Hapus baris jika kolom tertentu missing
df_clean = df.dropna(subset=['kolom_penting'])

# Hapus jika banyak kolom missing
df_clean = df.dropna(thresh=5)  # Keep hanya baris dengan >= 5 non-null values

# Parameters
df.dropna(
    axis=0,        # 0 untuk baris (default), 1 untuk kolom
    how='any',     # 'any' atau 'all' (hapus kalau ada any missing atau semua missing)
    subset=[...],  # Hanya check kolom tertentu
    thresh=None,   # Keep rows dengan >= thresh non-null values
    inplace=False  # Ubah original df atau return new df
)
```

### 2. Drop Columns (Penghapusan Kolom)

Hapus kolom yang punya banyak missing values.

**Kapan gunakan:**
- ✅ Kolom tidak penting
- ✅ Kolom punya banyak missing (> 50%)

```python
# Drop kolom dengan > 50% missing
threshold = 0.5
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df_clean = df.drop(columns=cols_to_drop)

# Drop specific kolom
df_clean = df.drop(['kolom1', 'kolom2'], axis=1)
```

### 3. Forward Fill & Backward Fill (Time Series)

Untuk time series data, bisa isi missing dengan nilai sebelumnya (ffill) atau sesudahnya (bfill).

**Kapan gunakan:**
- ✅ Time series data
- ✅ Data dengan order importance

```python
# Forward fill - isi dengan nilai sebelumnya
df['kolom'].fillna(method='ffill', inplace=True)

# Backward fill - isi dengan nilai sesudahnya
df['kolom'].fillna(method='bfill', inplace=True)

# Kombinasi: ffill kemudian bfill (untuk awal data)
df['kolom'].fillna(method='ffill', inplace=True)
df['kolom'].fillna(method='bfill', inplace=True)

# Limit propagation
df['kolom'].fillna(method='ffill', limit=3)  # Forward fill max 3 values
```

### 4. Constant Value Imputation

Isi missing dengan nilai konstan tertentu.

**Kapan gunakan:**
- ✅ Missing bermakna (misal: tidak ada pembelian = 0)
- ✅ Categorical data - bisa buat kategori "Unknown"

```python
# Isi dengan 0
df['kolom'].fillna(0, inplace=True)

# Isi dengan string tertentu
df['kategori'].fillna('Unknown', inplace=True)

# Isi dengan berbeda-beda per kolom
df.fillna({
    'age': 0,
    'name': 'Unknown',
    'score': -1
}, inplace=True)
```

### 5. Mean / Median Imputation

Isi missing dengan mean (rata-rata) atau median. Mean lebih sederhana, median lebih robust terhadap outliers.

**Kapan gunakan:**
- ✅ Numerik data normal distribution → gunakan mean
- ✅ Numerik data skewed distribution → gunakan median
- ✅ Missing sedikit (< 5%)

```python
# Mean imputation
mean_val = df['age'].mean()
df['age'].fillna(mean_val, inplace=True)

# Median imputation (lebih robust)
median_val = df['age'].median()
df['age'].fillna(median_val, inplace=True)

# Quick way
df['age'].fillna(df['age'].median(), inplace=True)

# Mode imputation (untuk kategorik)
df['kategori'].fillna(df['kategori'].mode()[0], inplace=True)
```

### 6. Group-based Imputation

Isi missing berdasarkan group/kategori - lebih akurat karena mempertimbangkan karakteristik grup.

**Kapan gunakan:**
- ✅ Data terstruktur dalam groups
- ✅ Berbeda pattern antar grup

```python
# Mean per group
df['age'] = df.groupby('gender')['age'].transform(
    lambda x: x.fillna(x.mean())
)

# Median per group
df['salary'] = df.groupby('department')['salary'].transform(
    lambda x: x.fillna(x.median())
)

# Mode per group
df['skill'] = df.groupby('department')['skill'].transform(
    lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
)

# Multiple groups
df['value'] = df.groupby(['category', 'subcategory'])['value'].transform(
    lambda x: x.fillna(x.median())
)
```

### 7. Interpolation

Linear interpolation - isi missing dengan nilai antar points.

**Kapan gunakan:**
- ✅ Time series atau sequential data
- ✅ Expected continuous relationship

```python
# Linear interpolation
df['kolom'].interpolate(method='linear', inplace=True)

# Polynomial interpolation
df['kolom'].interpolate(method='polynomial', order=2, inplace=True)

# Spline interpolation
df['kolom'].interpolate(method='spline', order=2, inplace=True)

# Fill limit
df['kolom'].interpolate(limit=2)  # Interpolate max 2 consecutive missing
```

---

## 📊 Strategi Selection Guide

Pilih strategi berdasarkan konteks:

| Persentase | Strategi | Kondisi |
| ---------- | -------- | ------- |
| < 5% | Drop baris | Missing random, data tidak penting |
| < 5% | Mean/Median | Numerik, missing random |
| 5-20% | Group-based | Numerik, ada group structure |
| 5-20% | Interpolation | Time series |
| > 20% | Forward/Backward fill | Time series dengan trend |
| > 50% | Drop kolom | Kolom tidak penting |

---

## ✏️ Praktik: Missing Value Handling

```python
import pandas as pd
import numpy as np

# Create sample data dengan missing values
np.random.seed(42)
data = {
    'id': range(1, 101),
    'age': np.random.randint(18, 65, 100),
    'salary': np.random.randint(30000, 150000, 100),
    'department': np.random.choice(['IT', 'HR', 'Finance'], 100),
    'score': np.random.uniform(0, 100, 100)
}

df = pd.DataFrame(data)

# Add missing values
df.loc[np.random.choice(100, 15, replace=False), 'age'] = np.nan
df.loc[np.random.choice(100, 10, replace=False), 'salary'] = np.nan
df.loc[np.random.choice(100, 5, replace=False), 'score'] = np.nan

print("=== BEFORE ===")
print(df.isnull().sum())

# Strategy 1: Drop age (15% missing)
df_strategy1 = df.dropna(subset=['age'])
print(f"\nStrategy 1 (Drop): {len(df)} → {len(df_strategy1)} rows")

# Strategy 2: Mean imputation untuk salary
df['salary'].fillna(df['salary'].mean(), inplace=True)

# Strategy 3: Group-based imputation untuk age
df['age'] = df.groupby('department')['age'].transform(
    lambda x: x.fillna(x.mean())
)

# Strategy 4: Forward fill untuk score
df['score'].fillna(method='ffill', inplace=True)

print("\n=== AFTER ===")
print(df.isnull().sum())
print("\nData integrity maintained ✓")
```

---

## 📝 Ringkasan Halaman Ini

### Missing Value Strategies

| Strategy | Pros | Cons | When |
| -------- | ---- | ---- | ---- |
| Drop | Simple, no assumption | Lose data | < 5% missing |
| Mean | Simple | Assume normality | Numeric, normal dist |
| Median | Robust outliers | Still assumption | Numeric, skewed dist |
| Group-based | More accurate | More complex | Grouped data |
| Forward Fill | Preserve trend | Bias | Time series |
| Interpolation | Smooth | Assume continuity | Sequential |

---

## ✏️ Latihan

### Latihan 1: Identify & Analyze

```python
df = pd.read_csv('data.csv')

# 1. Berapa missing values?
print(df.isnull().sum())

# 2. Persentase?
print((df.isnull().sum() / len(df) * 100).round(2))

# 3. Visualisasi
import matplotlib.pyplot as plt
(df.isnull().sum()[df.isnull().sum() > 0]).plot(kind='bar')
plt.show()
```

### Latihan 2: Handle Missing

```python
# 1. Drop rows
df_drop = df.dropna()

# 2. Mean imputation
df['age'].fillna(df['age'].mean(), inplace=True)

# 3. Group-based
df['salary'] = df.groupby('dept')['salary'].transform(
    lambda x: x.fillna(x.median())
)

# 4. Check hasil
print(df.isnull().sum())
```

### Latihan 3: Compare Strategies

```python
# Create multiple versions
df1 = df.dropna()  # Drop
df2 = df.copy()
df2['age'].fillna(df2['age'].mean(), inplace=True)  # Mean
df3 = df.copy()
df3['age'] = df3.groupby('category')['age'].transform(lambda x: x.fillna(x.mean()))  # Group

# Compare
print(f"Drop: {len(df1)} rows")
print(f"Mean: {len(df2)} rows")
print(f"Group-based: {len(df3)} rows")
```

---

## 🔗 Referensi

- [Pandas Missing Values](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Fillna Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)
