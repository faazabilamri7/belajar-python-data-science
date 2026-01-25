---
title: Pandas DataFrame Basics
description: Membuat dan mengeksplorasi DataFrame Pandas
sidebar:
  order: 5
---

## 🐼 DataFrame: Tabel Data 2D

### Apa itu DataFrame?

DataFrame adalah struktur data 2 dimensi (tabel dengan baris dan kolom), seperti:
- Spreadsheet Excel
- Tabel dalam database
- CSV file yang sudah di-load

**Karakteristik DataFrame:**
- 2 dimensi (rows dan columns)
- Setiap kolom adalah Series
- Fleksibel: berbagai tipe data di kolom berbeda
- Punya index untuk baris dan nama untuk kolom

---

## 📦 Membuat DataFrame

### Dari Dictionary

```python
import pandas as pd
import numpy as np

# Dictionary → DataFrame (key = column name)
data = {
    'nama': ['Andi', 'Budi', 'Citra', 'Diana', 'Eko'],
    'umur': [20, 21, 19, 22, 20],
    'jurusan': ['SI', 'TI', 'SI', 'TI', 'SI'],
    'ipk': [3.5, 3.7, 3.2, 3.8, 3.4]
}

df = pd.DataFrame(data)
print(df)
#     nama  umur jurusan  ipk
# 0   Andi    20      SI  3.5
# 1   Budi    21      TI  3.7
# 2  Citra    19      SI  3.2
# 3  Diana    22      TI  3.8
# 4    Eko    20      SI  3.4

# Dengan custom index
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D', 'E'])
print(df)
```

### Dari List of Lists/Dicts

```python
# List of lists
data = [
    ['Andi', 20, 'SI', 3.5],
    ['Budi', 21, 'TI', 3.7],
    ['Citra', 19, 'SI', 3.2]
]

df = pd.DataFrame(data, columns=['nama', 'umur', 'jurusan', 'ipk'])
print(df)

# List of dictionaries
data = [
    {'nama': 'Andi', 'umur': 20, 'ipk': 3.5},
    {'nama': 'Budi', 'umur': 21, 'ipk': 3.7},
    {'nama': 'Citra', 'umur': 19, 'ipk': 3.2}
]

df = pd.DataFrame(data)
print(df)
```

### Dari NumPy Array

```python
# NumPy array → DataFrame
arr = np.random.randn(5, 3)  # 5 rows, 3 columns
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print(df)
```

---

## 📊 Membaca File CSV

CSV (Comma-Separated Values) adalah format file data yang umum:

```python
# Read CSV dari file lokal
df = pd.read_csv('data.csv')

# Read dari URL
url = 'https://raw.githubusercontent.com/user/repo/data.csv'
df = pd.read_csv(url)

# Opsi umum
df = pd.read_csv('data.csv',
    sep=',',              # delimiter (default comma, bisa ';' atau '\t')
    header=0,            # baris ke berapa yang jadi header
    index_col=0,         # kolom ke berapa yang jadi index
    nrows=100,           # baca hanya 100 baris
    na_values=['NA', 'null', '-'],  # nilai apa yang dianggap NaN
    dtype={'umur': int, 'nama': str},  # specify data types
    encoding='utf-8'     # encoding file
)

# Skip rows
df = pd.read_csv('data.csv', skiprows=2)  # skip 2 baris pertama

# Specify column names
df = pd.read_csv('data.csv', names=['A', 'B', 'C', 'D'])
```

---

## 🔍 Eksplorasi Data Awal

Setelah load data, langkah pertama adalah eksplorasi:

```python
# Asumsi df sudah di-load
df = pd.read_csv('data.csv')

# 1. Lihat beberapa baris pertama
print(df.head())       # 5 baris pertama (default)
print(df.head(10))     # 10 baris pertama

# Lihat beberapa baris terakhir
print(df.tail())       # 5 baris terakhir
print(df.tail(3))      # 3 baris terakhir

# 2. Info dataset
print(df.shape)        # (rows, columns) → (100, 5)
print(df.columns)      # nama kolom
print(df.dtypes)       # tipe data setiap kolom

# 3. Ringkasan lengkap
print(df.info())
# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  -----  ------
#  0   nama    100    object
#  1   umur    100    int64
#  2   ipk     98     float64  ← ada 2 missing values
# dtypes: object, int64, float64, ...

# 4. Statistik deskriptif
print(df.describe())   # statistik untuk numeric columns
# Output:
#        umur        ipk
# count  100.0  98.000000
# mean    20.5   3.500000
# std      1.5   0.300000
# min     18.0   2.800000
# 25%     19.0   3.200000
# 50%     20.5   3.500000
# 75%     21.0   3.800000
# max     23.0   4.000000

# 5. Data types
print(df.dtypes)
# Output:
# nama        object
# umur         int64
# ipk        float64
# dtype: object
```

---

## 📋 Struktur dan Metadata DataFrame

```python
df = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra'],
    'umur': [20, 21, 19],
    'ipk': [3.5, 3.7, 3.2]
})

# Index dan columns
print(df.index)      # RangeIndex(start=0, stop=3, step=1)
print(df.columns)    # Index(['nama', 'umur', 'ipk'], dtype='object')

# Ubah index dan columns
df.index = ['A', 'B', 'C']
df.columns = ['nama_lengkap', 'usia', 'nilai_akhir']

# Rename specific columns
df = df.rename(columns={'nama': 'nama_lengkap', 'umur': 'usia'})

# Rename index
df = df.rename(index={0: 'siswa_1', 1: 'siswa_2'})

# Lowercase column names
df.columns = df.columns.str.lower()

# Replace spaces di column names
df.columns = df.columns.str.replace(' ', '_')

# Memory usage
print(df.memory_usage(deep=True))
# Output: bytes digunakan setiap kolom

# Sample (ambil random rows)
print(df.sample(2))   # 2 rows random
print(df.sample(frac=0.5))  # 50% rows random
```

---

## 🔍 Akses Data Dasar

### Mengakses Kolom

```python
df = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra'],
    'umur': [20, 21, 19],
    'ipk': [3.5, 3.7, 3.2]
})

# Akses satu kolom (return Series)
print(df['nama'])
print(df.nama)  # alternatif (hanya jika nama kolom valid identifier)

# Akses multiple kolom (return DataFrame)
print(df[['nama', 'ipk']])
print(df[['umur']])  # masih DataFrame (bukan Series)

# Kolom yang tidak ada → error
# print(df['gaji'])  # KeyError!
```

### Mengakses Baris dengan .loc[] dan .iloc[]

```python
# .loc[] - akses berdasarkan LABEL index
print(df.loc[0])        # baris dengan index 0 (Series)
print(df.loc[0:2])      # baris 0-2 inclusive
print(df.loc[0, 'nama']) # specific cell
print(df.loc[0:1, ['nama', 'umur']])  # subset

# .iloc[] - akses berdasarkan POSISI integer
print(df.iloc[0])       # baris pertama (posisi 0)
print(df.iloc[0:2])     # baris pertama dan kedua
print(df.iloc[0, 0])    # baris 0, kolom 0 (specific cell)
print(df.iloc[0:2, [0, 2]])  # baris 0-1, kolom 0 dan 2
```

### Conditional Selection

```python
# Boolean indexing
print(df[df['umur'] > 20])
# Output: hanya baris dengan umur > 20

# Multiple conditions
print(df[(df['umur'] > 19) & (df['ipk'] >= 3.5)])
print(df[(df['nama'] == 'Andi') | (df['umur'] > 20)])

# isin() - check membership
print(df[df['nama'].isin(['Andi', 'Citra'])])

# query() - more readable
print(df.query('umur > 20'))
print(df.query('umur > 20 and ipk >= 3.5'))
```

---

## 📊 Data Types Conversion

```python
df = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra'],
    'umur': ['20', '21', '19'],  # string, seharusnya int
    'ipk': [3.5, 3.7, 3.2]
})

print(df.dtypes)
# umur    object  ← harusnya int64

# Convert type
df['umur'] = df['umur'].astype(int)
print(df.dtypes)
# umur    int64  ✓

# Conversions common
df['umur'] = df['umur'].astype('int32')
df['ipk'] = df['ipk'].astype('float32')

# String to datetime
df['tanggal'] = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])

# Categorical (untuk data dengan kategori terbatas)
df['jurusan'] = df['jurusan'].astype('category')

# pd.to_numeric - convert dengan error handling
df['nilai'] = pd.to_numeric(df['nilai'], errors='coerce')
# errors='coerce': invalid parsing → NaN
# errors='ignore': invalid parsing → asli
# errors='raise': invalid parsing → exception
```

---

## 📝 Praktik: Load dan Eksplorasi

```python
import pandas as pd

# Buat sample data
data = {
    'nama': ['Andi', 'Budi', 'Citra', 'Diana', 'Eko'],
    'umur': [20, 21, 19, 22, 20],
    'jurusan': ['SI', 'TI', 'SI', 'TI', 'SI'],
    'ipk': [3.5, 3.7, 3.2, 3.8, 3.4],
    'sks': [120, 114, 108, 125, 110]
}

df = pd.DataFrame(data)

# Eksplorasi
print("=== Info Dasar ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Statistik Deskriptif ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== 5 Baris Pertama ===")
print(df.head())

print("\n=== Unique Values ===")
print(f"Jurusan: {df['jurusan'].unique()}")
print(f"Jumlah unique jurusan: {df['jurusan'].nunique()}")
```

---

## 📝 Ringkasan Halaman Ini

### DataFrame Creation & Exploration

| Operation | Contoh |
| --------- | ------ |
| Create | `pd.DataFrame(dict)`, `pd.DataFrame(list)` |
| Read CSV | `pd.read_csv('file.csv')` |
| Head/Tail | `df.head()`, `df.tail()` |
| Info | `df.info()`, `df.describe()` |
| Shape | `df.shape` |
| Columns | `df.columns` |
| Access | `df['col']`, `df.loc[0]`, `df.iloc[0]` |
| Filter | `df[df['col'] > 5]` |

---

## ✏️ Latihan

### Latihan 1: Create DataFrame

```python
# 1. Create dari dictionary
data = {
    'produk': ['Laptop', 'Mouse', 'Keyboard'],
    'harga': [10000000, 200000, 500000],
    'stok': [5, 50, 30]
}
df = pd.DataFrame(data)
print(df)

# 2. Add custom index
df.index = ['A', 'B', 'C']
print(df)

# 3. Get info
print(df.info())
```

### Latihan 2: Eksplorasi

```python
# Create sample data
df = pd.DataFrame({
    'nama': ['Faaza', 'Budi', 'Citra', 'Diana'],
    'nilai': [85, 90, 78, 92],
    'kelas': ['A', 'B', 'A', 'B']
})

# 1. Shape
print(f"Shape: {df.shape}")

# 2. Head/Tail
print(df.head(2))
print(df.tail(1))

# 3. Describe
print(df.describe())
```

### Latihan 3: Access & Filter

```python
df = pd.DataFrame({
    'nama': ['Faaza', 'Budi', 'Citra', 'Diana'],
    'umur': [25, 23, 22, 24],
    'gaji': [5000000, 4500000, 3500000, 5500000]
})

# 1. Access kolom
print(df['nama'])

# 2. Access baris
print(df.loc[1])
print(df.iloc[0])

# 3. Filter
print(df[df['gaji'] > 4000000])
print(df[df['umur'] >= 23])
```

---

## 🔗 Referensi

- [Pandas DataFrame Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [Pandas read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [Pandas User Guide - Basics](https://pandas.pydata.org/docs/user_guide/basics.html)
