---
title: EDA Fundamentals
description: Dasar-dasar Exploratory Data Analysis
sidebar:
  order: 2
---

## 🔍 Apa itu EDA?

**Exploratory Data Analysis (EDA)** adalah proses investigasi awal pada data untuk:

- 📊 Menemukan pola dan anomali
- 📈 Memahami distribusi data
- 🔗 Mengidentifikasi hubungan antar variabel
- 🎯 Memformulasikan hipotesis

### Mengapa EDA Penting?

> "Garbage In, Garbage Out" - Model ML hanya sebaik data yang digunakan

EDA membantu kita:
- ✅ Memahami data sebelum modeling
- ✅ Menemukan masalah dalam data (missing, outliers, errors)
- ✅ Menemukan insights penting
- ✅ Memilih variabel yang relevan
- ✅ Membuat keputusan tentang preprocessing

### EDA dalam CRISP-DM

```
Business Understanding → Data Understanding (EDA!) → Data Preparation → Modeling → Evaluation → Deployment
```

---

## 📊 Langkah 1: Load dan Inspeksi Data

Langkah pertama dalam EDA adalah meload data dan melakukan inspeksi awal.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (dari seaborn - contoh built-in)
df = sns.load_dataset('titanic')

# Atau dari file CSV:
# df = pd.read_csv('dataset.csv')

# Atau dari URL:
# df = pd.read_csv('https://url/dataset.csv')
```

### Inspeksi Awal

```python
# 1. Dimensi dataset
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
print(f"Shape: {df.shape}")

# 2. Nama kolom
print(f"\nNama kolom:\n{df.columns.tolist()}")

# 3. Tipe data setiap kolom
print(f"\nTipe data:\n{df.dtypes}")

# 4. Lihat beberapa baris pertama
print(f"\n5 Data Pertama:")
print(df.head())

# 5. Lihat beberapa baris terakhir
print(f"\n5 Data Terakhir:")
print(df.tail())

# 6. Random sample
print(f"\nRandom 3 baris:")
print(df.sample(3))
```

### Info Lengkap

```python
# Ringkasan lengkap
df.info()

# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 15 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       ------          -----
#  0   survived     891 non-null    int64
#  1   pclass       891 non-null    int64
#  2   sex          891 non-null    object
#  3   age          714 non-null    float64  ← 177 missing values
#  4   ...

# Memory usage
print(df.memory_usage(deep=True))
print(f"Total memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

---

## 📈 Langkah 2: Ringkasan Statistik

Statistik deskriptif memberikan informasi tentang central tendency, spread, dan distribusi data.

```python
# Statistik untuk kolom numerik
print(df.describe())

# Output:
#        survived       pclass         age       fare
# count  891.000000  891.000000  714.000000  891.000000
# mean     0.383838    2.308642   29.699118   32.204208
# std      0.486592    0.836071   14.526497   49.693429
# min      0.000000    1.000000    0.420000    0.000000
# 25%      0.000000    2.000000   20.125000    7.910400
# 50%      0.000000    3.000000   28.000000   14.454200
# 75%      1.000000    3.000000   38.000000   31.000000
# max      1.000000    3.000000   80.000000  512.329200

# Include categorical columns
print(df.describe(include='object'))

# Output:
#        sex embarked
# count  891       889
# unique   2         3
# top   male        S
# freq  577       644

# Include all columns
print(df.describe(include='all'))

# Custom percentiles
print(df.describe(percentiles=[0.1, 0.5, 0.9]))
```

### Statistik Individual

```python
# Untuk satu kolom
print(df['age'].describe())

# Statistik spesifik
print(f"Mean: {df['age'].mean():.2f}")
print(f"Median: {df['age'].median():.2f}")
print(f"Std Dev: {df['age'].std():.2f}")
print(f"Min: {df['age'].min():.2f}")
print(f"Max: {df['age'].max():.2f}")
print(f"Q1: {df['age'].quantile(0.25):.2f}")
print(f"Q3: {df['age'].quantile(0.75):.2f}")
print(f"IQR: {df['age'].quantile(0.75) - df['age'].quantile(0.25):.2f}")
```

---

## 🔢 Langkah 3: Unique Values dan Frekuensi

Untuk kolom kategorikal, penting untuk tahu berapa banyak unique values dan distribusinya.

```python
# Unique values
print(f"Unique values di 'sex': {df['sex'].nunique()}")
print(f"Values: {df['sex'].unique()}")

# Value counts (frekuensi)
print(df['sex'].value_counts())

# Output:
# male      577
# female    314
# Name: sex, dtype: int64

# Include NaN
print(df['sex'].value_counts(dropna=False))

# Normalized (persentase)
print(df['sex'].value_counts(normalize=True))

# Output:
# male      0.647486
# female    0.352514
# Name: sex, dtype: float64

# Sort by index (alphabetical)
print(df['sex'].value_counts().sort_index())
```

---

## ✏️ Missing Values Overview

Identifikasi ada missing values di mana dan berapa banyak.

```python
# Jumlah missing values per kolom
print(df.isnull().sum())

# Output:
# survived       0
# pclass         0
# sex            0
# age          177  ← 177 missing values
# sibsp          0
# parch          0
# fare           0
# embarked       2
# dtype: int64

# Persentase missing values
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0])

# Output:
# age        19.865470
# embarked    0.224467
# dtype: float64

# Total missing
print(f"Total missing: {df.isnull().sum().sum()}")

# Boolean mask
print(df.isnull())

# Baris dengan minimal satu missing
print(df[df.isnull().any(axis=1)].head())
```

---

## 📋 Template Inspeksi Cepat

Berikut adalah template untuk melakukan inspeksi data dengan cepat:

```python
def inspect_data(df):
    """Quick data inspection"""
    
    print("=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)
    
    # 1. Shape
    print(f"\n1. SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # 2. Columns
    print(f"\n2. COLUMNS:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col} ({df[col].dtype})")
    
    # 3. Data types
    print(f"\n3. DATA TYPES:")
    print(df.dtypes.value_counts())
    
    # 4. Missing values
    print(f"\n4. MISSING VALUES:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   No missing values!")
    
    # 5. Duplicates
    print(f"\n5. DUPLICATES: {df.duplicated().sum()} rows")
    
    # 6. Memory usage
    print(f"\n6. MEMORY: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 7. Numeric summary
    print(f"\n7. NUMERIC SUMMARY:")
    print(df.describe().loc[['mean', 'std', 'min', 'max']])
    
    # 8. Categorical summary
    print(f"\n8. CATEGORICAL SUMMARY:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols[:3]:  # Show first 3
        print(f"\n   {col}:")
        print(f"   {df[col].value_counts().head(3).to_dict()}")

# Gunakan
inspect_data(df)
```

---

## 🧮 Perbandingan Statistic untuk Group

Bandingkan statistik antar group dapat memberikan insight tentang differences antar kategori.

```python
# Statistik per group
print(df.groupby('sex')['age'].describe())

# Output:
#         count   mean   std   min   25%   50%   75%   max
# sex
# female   314.0  27.92 14.10  0.42  18.0  27.0  38.0  63.0
# male     577.0  30.73 14.88  0.42  20.0  29.0  40.0  80.0

# Multiple statistics
print(df.groupby('pclass').agg({
    'age': ['count', 'mean', 'std', 'min', 'max'],
    'fare': ['mean', 'median'],
    'survived': 'mean'
}))

# Counts per group
print(df['embarked'].value_counts())
print(df.groupby(['sex', 'pclass']).size())
```

---

## 📝 Ringkasan Halaman Ini

### EDA Steps

| Step | Description | Tools |
| ---- | ----------- | ----- |
| 1. Load | Baca data dari file/URL | `pd.read_csv()` |
| 2. Inspect | Lihat shape, columns, dtypes | `df.info()`, `df.head()` |
| 3. Statistics | Hitung mean, std, quantiles | `df.describe()` |
| 4. Missing | Cek missing values | `df.isnull().sum()` |
| 5. Unique | Check unique & frequency | `value_counts()` |

---

## ✏️ Latihan

### Latihan 1: Load & Inspect

```python
import pandas as pd
import seaborn as sns

# Load dataset
df = sns.load_dataset('iris')

# 1. Shape?
print(f"Shape: {df.shape}")

# 2. Columns dan types?
print(df.dtypes)

# 3. First 3 rows?
print(df.head(3))

# 4. Info lengkap?
df.info()
```

### Latihan 2: Statistik

```python
# 1. Summary statistics
print(df.describe())

# 2. Mean per species
print(df.groupby('species')['sepal_length'].mean())

# 3. Count per species
print(df['species'].value_counts())
```

### Latihan 3: Missing & Duplicates

```python
# 1. Ada missing?
print(df.isnull().sum())

# 2. Ada duplicate?
print(df.duplicated().sum())

# 3. Baris yang unique?
print(df.drop_duplicates())
```

---

## 🔗 Referensi

- [Pandas Describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
- [Pandas Info](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html)
- [Pandas Value Counts](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html)
