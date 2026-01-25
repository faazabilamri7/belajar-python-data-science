---
title: Pandas Series
description: Memahami dan memanipulasi Pandas Series
sidebar:
  order: 4
---

## 🐼 Pandas Series Fundamentals

### Apa itu Series?

Series adalah struktur data 1 dimensi (seperti kolom dalam spreadsheet atau array dengan label).

**Karakteristik Series:**
- 1 dimensi
- Setiap elemen punya label (index)
- Index bisa custom atau default (0, 1, 2, ...)
- Bisa berisi berbagai tipe data

### Membuat Series

```python
import pandas as pd
import numpy as np

# Dari list dengan default index (0, 1, 2, ...)
s1 = pd.Series([10, 20, 30, 40, 50])
print(s1)
# 0    10
# 1    20
# 2    30
# 3    40
# 4    50
# dtype: int64

# Dari list dengan custom index
nilai_ujian = pd.Series(
    [85, 90, 78, 92, 88],
    index=['Matematika', 'Fisika', 'Kimia', 'Biologi', 'Bahasa']
)
print(nilai_ujian)
# Matematika    85
# Fisika        90
# Kimia         78
# Biologi       92
# Bahasa        88
# dtype: int64

# Dari dictionary (key menjadi index)
data = {
    'Andi': 85,
    'Budi': 90,
    'Citra': 78,
    'Diana': 92
}
s2 = pd.Series(data)
print(s2)
# Andi     85
# Budi     90
# Citra    78
# Diana    92
# dtype: int64

# Dari NumPy array
arr = np.array([10, 20, 30, 40])
s3 = pd.Series(arr, index=['a', 'b', 'c', 'd'])
print(s3)
# a    10
# b    20
# c    30
# d    40
# dtype: int64

# Series dengan tipe data berbeda
s4 = pd.Series([10, 'text', 3.14, True])
print(s4)  # dtype: object (mixed types)
```

---

## 📊 Atribut Series

```python
nilai = pd.Series(
    [85, 90, 78, 92, 88],
    index=['Matematika', 'Fisika', 'Kimia', 'Biologi', 'Bahasa']
)

# Index dan values
print(nilai.index)    # Index(['Matematika', 'Fisika', ...])
print(nilai.values)   # array([85, 90, 78, 92, 88])

# Name
nilai.name = 'Nilai Ujian'
print(nilai.name)

# Shape dan size
print(nilai.shape)    # (5,)
print(nilai.size)     # 5

# Data type
print(nilai.dtype)    # int64

# Is unique
print(nilai.is_unique)  # True

# Informasi index
print(nilai.index.name)  # None (bisa di-set)
nilai.index.name = 'Mata Kuliah'
print(nilai)
```

---

## 🔍 Mengakses Elemen Series

### Indexing

```python
nilai = pd.Series(
    [85, 90, 78, 92, 88],
    index=['Matematika', 'Fisika', 'Kimia', 'Biologi', 'Bahasa']
)

# Akses dengan label (index)
print(nilai['Matematika'])   # 85
print(nilai['Fisika'])       # 90

# Akses dengan posisi (integer location)
print(nilai.iloc[0])         # 85 (elemen pertama)
print(nilai.iloc[1])         # 90 (elemen kedua)

# Akses dengan .loc[]
print(nilai.loc['Matematika'])  # 85

# Multiple elements dengan list of labels
print(nilai[['Matematika', 'Fisika']])
# Matematika    85
# Fisika        90

# Multiple elements dengan list of positions
print(nilai.iloc[[0, 2, 4]])
# Matematika    85
# Kimia         78
# Bahasa        88
```

### Slicing

```python
s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])

# Slicing dengan label (inclusive end!)
print(s['b':'d'])
# b    20
# c    30
# d    40

# Slicing dengan posisi (exclusive end)
print(s.iloc[1:4])
# b    20
# c    30
# d    40

# Slicing dengan kondisi
print(s[s > 25])
# c    30
# d    40
# e    50
```

---

## ✏️ Memodifikasi Series

```python
s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])

# Mengubah nilai individual
s['a'] = 15
print(s)

# Mengubah multiple values
s[['b', 'c']] = 100
print(s)

# Mengubah dengan kondisi
s[s < 30] = 0
print(s)

# Append (menambah elemen) - create Series baru
s_new = pd.Series([60], index=['f'])
s = pd.concat([s, s_new])
print(s)

# Drop (menghapus) - create Series baru
s_drop = s.drop('a')
print(s_drop)

# Rename index
s.index = ['A', 'B', 'C', 'D', 'E', 'F']
print(s)
```

---

## 📈 Operasi Matematika pada Series

Element-wise operations seperti di NumPy:

```python
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([10, 20, 30, 40, 50])

# Aritmatika
print(s1 + s2)      # [11 22 33 44 55]
print(s1 * 2)       # [2 4 6 8 10]
print(s2 / s1)      # [10 10 10 10 10]

# Fungsi matematika
print(np.sqrt(s1))  # [1. 1.41 1.73 2. 2.24]
print(np.exp(s1))   # [2.72 7.39 20.09 54.60 148.41]

# Comparison (return boolean Series)
print(s1 > 2)       # [False False True True True]
print(s1 == 3)      # [False False True False False]

# Filtering
print(s1[s1 > 2])   # [3 4 5]
```

---

## 📊 Operasi Agregasi pada Series

```python
nilai = pd.Series([85, 90, 78, 92, 88, 95, 76])

# Statistik dasar
print(nilai.sum())      # 604 - jumlah
print(nilai.mean())     # 86.29 - rata-rata
print(nilai.median())   # 88.0 - median
print(nilai.std())      # 6.66 - standard deviation
print(nilai.var())      # 44.39 - variance
print(nilai.min())      # 76 - minimum
print(nilai.max())      # 95 - maximum

# Counts dan unique
print(nilai.count())    # 7 - jumlah elemen (non-null)
print(nilai.nunique())  # 7 - jumlah unique values

# Quantiles
print(nilai.quantile(0.25))  # 0.25 quantile (Q1)
print(nilai.quantile(0.5))   # 0.5 quantile (median)
print(nilai.quantile(0.75))  # 0.75 quantile (Q3)

# Value counts (frekuensi)
s = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'])
print(s.value_counts())
# A    3
# B    2
# C    1

# Describe (summary statistics)
print(nilai.describe())
# count     7.00
# mean     86.29
# std       6.66
# min      76.00
# 25%      80.50
# 50%      88.00
# 75%      91.50
# max      95.00
```

---

## 🔄 Series dengan Missing Values

```python
import numpy as np

# Create Series dengan NaN (missing values)
s = pd.Series([1, 2, np.nan, 4, None, 6])
print(s)
# 0    1.0
# 1    2.0
# 2    NaN
# 3    4.0
# 4    NaN
# 5    6.0

# Check missing values
print(s.isnull())      # [False False True False True False]
print(s.notnull())     # [True True False True False True]
print(s.isnull().sum()) # 2 - jumlah missing values

# Drop missing values
s_clean = s.dropna()
print(s_clean)
# 0    1.0
# 1    2.0
# 3    4.0
# 5    6.0

# Fill missing values
s_filled = s.fillna(0)      # fill dengan 0
print(s_filled)

s_filled = s.fillna(s.mean())  # fill dengan mean
print(s_filled)

# Forward fill (ffill) dan backward fill (bfill)
s2 = pd.Series([1, np.nan, np.nan, 4, 5])
print(s2.ffill())   # [1 1 1 4 5] - propagate last value forward
print(s2.bfill())   # [1 4 4 4 5] - propagate next value backward
```

---

## 🧮 String Operations pada Series

Jika Series berisi string:

```python
s = pd.Series(['hello', 'world', 'python', 'data'])

# String methods (akses via .str)
print(s.str.upper())           # ['HELLO' 'WORLD' 'PYTHON' 'DATA']
print(s.str.len())             # [5 5 6 4]
print(s.str.startswith('p'))   # [False False True False]
print(s.str.contains('a'))     # [False False True True]
print(s.str.replace('o', '0')) # ['hell0' 'w0rld' 'pyth0n' 'data']
print(s.str.split('l'))        # Series of lists

# Extract substring
s2 = pd.Series(['John_25', 'Jane_30', 'Bob_22'])
print(s2.str.extract(r'(\w+)_(\d+)', expand=True))
#     0   1
# 0  John  25
# 1  Jane  30
# 2   Bob  22
```

---

## 📝 Praktik: Data Processing

```python
import pandas as pd
import numpy as np

# Simulasi data nilai siswa
np.random.seed(42)
nilai_raw = pd.Series(
    np.random.normal(75, 15, 50),
    index=[f'Siswa_{i}' for i in range(1, 51)]
)

# Clip ke range 0-100
nilai = nilai_raw.clip(0, 100)

# 1. Statistik dasar
print("=== Statistik Nilai ===")
print(nilai.describe())

# 2. Grade assignment
def get_grade(score):
    if score >= 85:
        return 'A'
    elif score >= 75:
        return 'B'
    elif score >= 65:
        return 'C'
    elif score >= 55:
        return 'D'
    else:
        return 'E'

grades = nilai.apply(get_grade)
print("\n=== Distribusi Grade ===")
print(grades.value_counts().sort_index())

# 3. Top 5 siswa
print("\n=== Top 5 Nilai Tertinggi ===")
print(nilai.nlargest(5))

# 4. Bottom 5 siswa
print("\n=== 5 Nilai Terendah ===")
print(nilai.nsmallest(5))
```

---

## 📝 Ringkasan Halaman Ini

### Series Operations

| Operation | Contoh |
| --------- | ------ |
| Create | `pd.Series([1,2,3])`, `pd.Series(dict)` |
| Access | `s['label']`, `s.iloc[0]`, `s[['a','b']]` |
| Slice | `s['a':'c']`, `s.iloc[1:4]` |
| Math | `s + 10`, `s * 2`, `np.sqrt(s)` |
| Aggregate | `s.sum()`, `s.mean()`, `s.describe()` |
| Modify | `s['a'] = 5`, `s.fillna(0)` |
| Filter | `s[s > 10]`, `s.isnull()` |

---

## ✏️ Latihan

### Latihan 1: Create and Access

```python
# 1. Create Series dari list
s1 = pd.Series([10, 20, 30, 40, 50])

# 2. Create Series dari dictionary
data = {'A': 100, 'B': 200, 'C': 150}
s2 = pd.Series(data)

# 3. Access elemen
print(s2['A'])
print(s2[['A', 'C']])

# 4. Slice
print(s1[1:4])
```

### Latihan 2: Aggregation

```python
scores = pd.Series([85, 90, 78, 92, 88, 95, 76, 82])

# 1. Basic stats
print(f"Mean: {scores.mean():.2f}")
print(f"Median: {scores.median():.2f}")
print(f"Std: {scores.std():.2f}")

# 2. Min-Max
print(f"Min: {scores.min()}, Max: {scores.max()}")

# 3. Passing rate (>= 80)
passing = (scores >= 80).sum()
total = len(scores)
print(f"Passing rate: {passing}/{total} = {(passing/total)*100:.1f}%")
```

### Latihan 3: Missing Values

```python
import numpy as np

# Create Series dengan NaN
s = pd.Series([10, 20, np.nan, 40, None, 60])

# 1. Find missing values
print(f"Missing count: {s.isnull().sum()}")

# 2. Drop NaN
s_clean = s.dropna()
print(s_clean)

# 3. Fill with mean
s_filled = s.fillna(s.mean())
print(s_filled)
```

---

## 🔗 Referensi

- [Pandas Series Documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.html)
- [Pandas User Guide - Series](https://pandas.pydata.org/docs/user_guide/basics.html)
