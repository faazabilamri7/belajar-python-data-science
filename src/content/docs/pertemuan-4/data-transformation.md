---
title: Data Transformation
description: Teknik transformasi dan feature engineering pada data
sidebar:
  order: 6
---

## 🔧 Data Transformation

Transformasi data mengubah format, struktur, atau value dari data untuk membuat data lebih siap untuk analisis dan modeling.

---

## 📝 1. Type Casting

Memastikan setiap kolom memiliki tipe data yang benar.

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

# Cek tipe data saat ini
print(df.dtypes)

# String ke Numeric
df['harga'] = pd.to_numeric(df['harga'], errors='coerce')
# errors='coerce': invalid → NaN
# errors='ignore': invalid → original
# errors='raise': invalid → exception

# Numeric ke String
df['kode'] = df['kode'].astype(str)

# String ke DateTime
df['tanggal'] = pd.to_datetime(df['tanggal'])
# Formats:
# df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d/%m/%Y')
# df['tanggal'] = pd.to_datetime(df['tanggal'], format='%Y-%m-%d')

# Integer ke Float
df['nilai'] = df['nilai'].astype(float)

# Category (hemat memory)
df['kategori'] = df['kategori'].astype('category')

# Boolean
df['is_active'] = df['is_active'].astype(bool)

# Check berhasil
print(df.dtypes)
```

---

## 📅 2. Feature Engineering dari DateTime

DateTime columns bisa memberikan features tambahan yang useful untuk modeling.

```python
# Pastikan column adalah datetime
df['tanggal'] = pd.to_datetime(df['tanggal'])

# Extract komponen
df['tahun'] = df['tanggal'].dt.year
df['bulan'] = df['tanggal'].dt.month
df['hari'] = df['tanggal'].dt.day
df['hari_minggu'] = df['tanggal'].dt.dayofweek  # 0=Senin, 6=Minggu
df['nama_hari'] = df['tanggal'].dt.day_name()   # 'Monday', 'Tuesday', ...
df['kuartal'] = df['tanggal'].dt.quarter
df['minggu'] = df['tanggal'].dt.isocalendar().week

# Boolean features
df['is_weekend'] = df['tanggal'].dt.dayofweek >= 5
df['is_akhir_bulan'] = df['tanggal'].dt.day > 25

# Cyclical features (untuk weekday, bulan, etc - untuk model)
# Mengkonversi cyclical data menjadi sin/cos untuk preserve distance
df['bulan_sin'] = np.sin(2 * np.pi * df['bulan'] / 12)
df['bulan_cos'] = np.cos(2 * np.pi * df['bulan'] / 12)

# Time-based features
df['hari_dalam_tahun'] = df['tanggal'].dt.dayofyear
df['minggu_dalam_tahun'] = df['tanggal'].dt.isocalendar().week
df['hari_sejak_awal'] = (df['tanggal'] - df['tanggal'].min()).dt.days

# Multiple dates
df['tanggal_mulai'] = pd.to_datetime(df['tanggal_mulai'])
df['tanggal_selesai'] = pd.to_datetime(df['tanggal_selesai'])
df['durasi_hari'] = (df['tanggal_selesai'] - df['tanggal_mulai']).dt.days
```

---

## 🧹 3. String Cleaning

Text data sering punya inconsistencies yang harus di-clean.

```python
# Strip whitespace
df['nama'] = df['nama'].str.strip()

# Standardize case
df['nama'] = df['nama'].str.upper()         # UPPERCASE
df['nama'] = df['nama'].str.lower()         # lowercase
df['nama'] = df['nama'].str.title()         # Title Case

# Remove spasi berlebih
df['alamat'] = df['alamat'].str.replace(r'\s+', ' ', regex=True)

# Replace karakter
df['telepon'] = df['telepon'].str.replace('-', '')
df['telepon'] = df['telepon'].str.replace(' ', '')

# Remove spesifik substring
df['kode'] = df['kode'].str.replace('ID-', '')

# Extract substring
df['nama_pertama'] = df['nama'].str.split(' ').str[0]
df['nama_akhir'] = df['nama'].str.split(' ').str[-1]

# Extract dengan regex
df['kode_pos'] = df['alamat'].str.extract(r'(\d{5})')
df['kota'] = df['alamat'].str.extract(r'(Jakarta|Bandung|Surabaya)')

# Check membership
df['is_gmail'] = df['email'].str.contains('gmail', case=False)
df['is_valid_email'] = df['email'].str.contains(r'\w+@\w+\.\w+', regex=True)

# String functions
df['nama_len'] = df['nama'].str.len()
df['has_space'] = df['nama'].str.contains(' ')
```

---

## 📊 4. Normalization & Standardization

Scaling numerical data ke range tertentu untuk modeling.

```python
# Min-Max Normalization (0-1 range)
df['harga_norm'] = (df['harga'] - df['harga'].min()) / (df['harga'].max() - df['harga'].min())
# Formula: (x - min) / (max - min)

# Standardization / Z-score (mean=0, std=1)
df['harga_std'] = (df['harga'] - df['harga'].mean()) / df['harga'].std()
# Formula: (x - mean) / std

# Robust scaling (menggunakan median & IQR - robust outliers)
Q1 = df['harga'].quantile(0.25)
Q3 = df['harga'].quantile(0.75)
IQR = Q3 - Q1
df['harga_robust'] = (df['harga'] - df['harga'].median()) / IQR

# Scikit-learn (reusable untuk test set)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Min-Max
scaler = MinMaxScaler()
df['harga_norm'] = scaler.fit_transform(df[['harga']])

# Standardization
scaler = StandardScaler()
df['harga_std'] = scaler.fit_transform(df[['harga']])

# Robust
scaler = RobustScaler()
df['harga_robust'] = scaler.fit_transform(df[['harga']])
```

---

## 📍 5. Binning (Kategorisasi Numerik)

Mengkonversi continuous numerik menjadi categorical untuk pattern discovery atau untuk models yang lebih cocok dengan categorical input.

### Equal-Width Binning

```python
# Cut - interval sama width
df['umur_grup'] = pd.cut(
    df['umur'],
    bins=[0, 18, 35, 50, 100],
    labels=['Anak', 'Dewasa Muda', 'Dewasa', 'Lansia'],
    include_lowest=True
)

# Dengan default labels
df['income_level'] = pd.cut(df['income'], bins=5)
# Labels: (0, 20000], (20000, 40000], ...

# Custom number of bins
df['nilai_grup'] = pd.cut(df['nilai'], bins=4)
```

### Equal-Frequency Binning (Quantile)

```python
# Qcut - kuantil sama (balanced groups)
df['income_quartile'] = pd.qcut(
    df['income'],
    q=4,
    labels=['Q1 (Terendah)', 'Q2', 'Q3', 'Q4 (Tertinggi)']
)

# Check distribusi (setiap group punya roughly sama rows)
print(df['income_quartile'].value_counts().sort_index())
```

---

## 🔢 6. Encoding Kategorik

Mengkonversi categorical data menjadi numerical untuk ML models.

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

# Manual
mapping = {'A': 0, 'B': 1, 'C': 2}
df['kategori_encoded'] = df['kategori'].map(mapping)

# Scikit-learn
le = LabelEncoder()
df['kategori_encoded'] = le.fit_transform(df['kategori'])

# Get mapping
for i, label in enumerate(le.classes_):
    print(f"{i}: {label}")
```

### One-Hot Encoding

```python
# Pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['kategori'], prefix='kategori', drop_first=False)

# Drop first untuk избежать multicollinearity
df_encoded = pd.get_dummies(df, columns=['kategori'], prefix='kategori', drop_first=True)

# Scikit-learn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['kategori']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
```

### Target Encoding (Mean Encoding)

```python
# Mean target value per category
target_encoding = df.groupby('kategori')['target'].mean()
df['kategori_target_encoded'] = df['kategori'].map(target_encoding)
```

---

## ➕ 7. Feature Creation

Membuat features baru dari kombinasi existing features.

```python
# Arithmetic operations
df['total'] = df['harga'] * df['quantity']
df['discount_amount'] = df['harga'] * df['discount_pct'] / 100

# Interactions
df['price_qty_interaction'] = df['harga'] * df['quantity']

# Ratios
df['ratio'] = df['nilai1'] / (df['nilai2'] + 1)  # +1 untuk avoid division by zero

# Log features
df['harga_log'] = np.log1p(df['harga'])
df['qty_log'] = np.log1p(df['quantity'])

# Polynomial features
df['harga_squared'] = df['harga'] ** 2
df['harga_sqrt'] = np.sqrt(df['harga'])

# Domain-specific
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100])
df['has_discount'] = (df['discount_pct'] > 0).astype(int)
```

---

## 📋 Transformation Pipeline Template

```python
def transform_data(df):
    """Complete data transformation pipeline"""
    
    df_transformed = df.copy()
    
    # 1. Type casting
    df_transformed['tanggal'] = pd.to_datetime(df_transformed['tanggal'])
    df_transformed['harga'] = pd.to_numeric(df_transformed['harga'], errors='coerce')
    
    # 2. DateTime features
    df_transformed['tahun'] = df_transformed['tanggal'].dt.year
    df_transformed['bulan'] = df_transformed['tanggal'].dt.month
    df_transformed['hari_minggu'] = df_transformed['tanggal'].dt.dayofweek
    df_transformed['is_weekend'] = df_transformed['hari_minggu'] >= 5
    
    # 3. String cleaning
    df_transformed['nama'] = df_transformed['nama'].str.strip().str.lower()
    
    # 4. Binning
    df_transformed['umur_group'] = pd.cut(df_transformed['umur'], 
                                           bins=[0, 18, 35, 50, 100],
                                           labels=['Anak', 'Dewasa Muda', 'Dewasa', 'Lansia'])
    
    # 5. Encoding
    df_transformed = pd.get_dummies(df_transformed, columns=['kategori'], drop_first=True)
    
    # 6. Feature creation
    df_transformed['total'] = df_transformed['harga'] * df_transformed['quantity']
    
    # 7. Normalization
    df_transformed['harga_norm'] = (df_transformed['harga'] - df_transformed['harga'].min()) / \
                                    (df_transformed['harga'].max() - df_transformed['harga'].min())
    
    return df_transformed

# Usage
df_clean = transform_data(df)
```

---

## 📝 Ringkasan Halaman Ini

### Data Transformation Techniques

| Technique | Purpose | Example |
| --------- | ------- | ------- |
| Type Casting | Correct data types | String → DateTime |
| Feature Engineering | Extract useful info | DateTime → Year, Month |
| String Cleaning | Standardize text | UPPERCASE, remove spaces |
| Normalization | Scale to range | 0-1 range |
| Standardization | Mean=0, Std=1 | Z-score normalization |
| Binning | Categorize numeric | Age → Age groups |
| Encoding | Numeric for models | One-hot encoding |

---

## ✏️ Latihan

### Latihan 1: Type Casting & DateTime

```python
df = pd.DataFrame({
    'tanggal': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'harga': ['100', '200', '300']
})

# 1. Convert types
df['tanggal'] = pd.to_datetime(df['tanggal'])
df['harga'] = pd.to_numeric(df['harga'])

# 2. Extract datetime features
df['tahun'] = df['tanggal'].dt.year
df['bulan'] = df['tanggal'].dt.month
df['hari'] = df['tanggal'].dt.day
```

### Latihan 2: String Cleaning & Binning

```python
df = pd.DataFrame({
    'nama': ['  JOHN  ', '  jane  ', '  BOB  '],
    'umur': [25, 35, 45]
})

# 1. Clean strings
df['nama'] = df['nama'].str.strip().str.title()

# 2. Bin numeric
df['umur_group'] = pd.cut(df['umur'], bins=[0, 30, 40, 50], 
                           labels=['Young', 'Mid', 'Senior'])
```

### Latihan 3: Encoding & Feature Creation

```python
df = pd.DataFrame({
    'kategori': ['A', 'B', 'A', 'C'],
    'nilai1': [10, 20, 15, 25],
    'nilai2': [5, 10, 8, 12]
})

# 1. One-hot encoding
df_encoded = pd.get_dummies(df, columns=['kategori'], drop_first=True)

# 2. Feature creation
df_encoded['interaction'] = df['nilai1'] * df['nilai2']
df_encoded['ratio'] = df['nilai1'] / (df['nilai2'] + 1)
```

---

## 🔗 Referensi

- [Pandas String Methods](https://pandas.pydata.org/docs/user_guide/text.html)
- [Pandas DateTime](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
