---
title: Data Operations & Joining
description: Operasi agregasi, joining, dan menyimpan data
sidebar:
  order: 7
---

## 📊 Operasi Agregasi Lanjut

### Aggregation Functions

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'tanggal': pd.date_range('2024-01-01', periods=5),
    'penjualan': [100, 150, 120, 200, 180],
    'biaya': [40, 60, 50, 80, 70],
    'kategori': ['A', 'B', 'A', 'B', 'A']
})

# Aggregasi single nilai
print(df['penjualan'].sum())      # 750
print(df['penjualan'].mean())     # 150
print(df['penjualan'].median())   # 150
print(df['penjualan'].std())      # 41.83
print(df['penjualan'].min())      # 100
print(df['penjualan'].max())      # 200
print(df['penjualan'].count())    # 5
print(df['penjualan'].nunique())  # 5

# Multiple aggregation
df.agg({
    'penjualan': ['sum', 'mean', 'std'],
    'biaya': ['min', 'max']
})

# Named aggregation
df.agg(
    total_penjualan=('penjualan', 'sum'),
    rata_penjualan=('penjualan', 'mean'),
    max_biaya=('biaya', 'max')
)
```

### Cumulative Functions

```python
arr = np.array([1, 2, 3, 4, 5])

# Cumulative sum
print(np.cumsum(arr))      # [1 3 6 10 15]

# Cumulative product
print(np.cumprod(arr))     # [1 2 6 24 120]

# Di Pandas DataFrame
df['cumsum'] = df['penjualan'].cumsum()
df['cumprod'] = df['biaya'].cumprod()

# Percentage change
df['pct_change'] = df['penjualan'].pct_change()
```

### Unique dan Value Counts

```python
df = pd.DataFrame({
    'kategori': ['A', 'B', 'A', 'C', 'B', 'A', 'D'],
    'nilai': [10, 20, 15, 30, 25, 12, 35]
})

# Unique values
print(df['kategori'].unique())      # ['A' 'B' 'C' 'D']
print(df['kategori'].nunique())     # 4

# Value counts (frekuensi)
print(df['kategori'].value_counts())
# A    3
# B    2
# C    1
# D    1
# Name: kategori, dtype: int64

# Sorted
print(df['kategori'].value_counts(sort=True))

# Include NaN
print(df['kategori'].value_counts(dropna=False))

# Normalize (percentage)
print(df['kategori'].value_counts(normalize=True))
# A    0.428571 (42.86%)
# B    0.285714
# C    0.142857
# D    0.142857
```

---

## 📊 Advanced GroupBy Operations

### Multi-level GroupBy

```python
df = pd.DataFrame({
    'tahun': [2023, 2023, 2023, 2024, 2024, 2024],
    'kuartal': [1, 1, 2, 1, 1, 2],
    'penjualan': [100, 150, 120, 200, 180, 220],
    'biaya': [40, 60, 50, 80, 70, 90]
})

# Group by multiple columns
result = df.groupby(['tahun', 'kuartal']).agg({
    'penjualan': 'sum',
    'biaya': 'mean'
})
print(result)

# Unstack (pivot result)
result_pivot = result.unstack(level=-1)
print(result_pivot)
```

### Transform vs Apply

```python
df = pd.DataFrame({
    'grup': ['A', 'A', 'A', 'B', 'B', 'B'],
    'nilai': [1, 2, 3, 4, 5, 6]
})

# GroupBy + mean (reduce to 1 value per group)
print(df.groupby('grup')['nilai'].mean())
# A    2.0
# B    5.0

# GroupBy + transform (broadcast back to original shape)
df['nilai_norm'] = df.groupby('grup')['nilai'].transform(lambda x: x - x.mean())
print(df)
# nilai_norm untuk A group: [1-2, 2-2, 3-2] = [-1, 0, 1]
# nilai_norm untuk B group: [4-5, 5-5, 6-5] = [-1, 0, 1]
```

### Filter pada GroupBy

```python
# Keep groups dengan count >= 2
df_filtered = df.groupby('grup').filter(lambda x: len(x) >= 2)
print(df_filtered)

# Keep groups dengan mean > 3
df_filtered = df.groupby('grup')['nilai'].filter(lambda x: x.mean() > 3)
print(df_filtered)
```

---

## 🔗 Joining dan Merging Advanced

### Different Join Types

```python
left = pd.DataFrame({
    'id': [1, 2, 3],
    'nama': ['Andi', 'Budi', 'Citra']
})

right = pd.DataFrame({
    'id': [2, 3, 4],
    'nilai': [90, 85, 95]
})

print("=== INNER JOIN ===")
# Hanya data yang ada di kedua tabel
inner = pd.merge(left, right, on='id', how='inner')
print(inner)
#   id  nama  nilai
# 0  2  Budi     90
# 1  3  Citra    85

print("\n=== LEFT JOIN ===")
# Semua data dari left, match dari right (atau NaN)
left_join = pd.merge(left, right, on='id', how='left')
print(left_join)
#   id   nama  nilai
# 0  1   Andi    NaN
# 1  2   Budi   90.0
# 2  3  Citra   85.0

print("\n=== RIGHT JOIN ===")
# Semua data dari right, match dari left (atau NaN)
right_join = pd.merge(left, right, on='id', how='right')
print(right_join)
#   id   nama  nilai
# 0  2   Budi     90
# 1  3  Citra     85
# 2  4    NaN     95

print("\n=== OUTER JOIN ===")
# Semua data dari kedua tabel
outer = pd.merge(left, right, on='id', how='outer')
print(outer)
#   id   nama  nilai
# 0  1   Andi    NaN
# 1  2   Budi   90.0
# 2  3  Citra   85.0
# 3  4    NaN   95.0
```

### Merge dengan Multiple Keys

```python
left = pd.DataFrame({
    'tahun': [2023, 2023, 2024],
    'bulan': [1, 2, 1],
    'penjualan': [100, 120, 150]
})

right = pd.DataFrame({
    'tahun': [2023, 2023, 2024],
    'bulan': [1, 2, 1],
    'biaya': [40, 50, 60]
})

# Merge dengan multiple keys
result = pd.merge(left, right, on=['tahun', 'bulan'])
print(result)
```

### Merge dengan Different Key Names

```python
df1 = pd.DataFrame({
    'id_siswa': [1, 2, 3],
    'nama': ['Andi', 'Budi', 'Citra']
})

df2 = pd.DataFrame({
    'siswa_id': [1, 2, 3],
    'nilai': [85, 90, 78]
})

# Merge dengan left_on, right_on
result = pd.merge(df1, df2, left_on='id_siswa', right_on='siswa_id')
print(result)
```

### Join by Index

```python
df1 = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra']
}, index=['A', 'B', 'C'])

df2 = pd.DataFrame({
    'nilai': [85, 90, 78]
}, index=['A', 'B', 'C'])

# Join by index
result = df1.join(df2)
print(result)
#       nama  nilai
# A     Andi     85
# B     Budi     90
# C   Citra     78
```

---

## 💾 Menyimpan Data

### Export ke File

```python
df = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra'],
    'nilai': [85, 90, 78]
})

# Save ke CSV
df.to_csv('data.csv', index=False)
# index=False: jangan save index column

# Save ke Excel
df.to_excel('data.xlsx', sheet_name='Sheet1', index=False)

# Save ke JSON
df.to_json('data.json', orient='records')

# Save ke Parquet (binary format, lebih cepat)
df.to_parquet('data.parquet')

# Save ke SQL database
# df.to_sql('tabel_nama', con=connection, if_exists='replace')
# if_exists: 'fail', 'replace', 'append'

# Save ke HTML
df.to_html('data.html')

# Save multiple sheets Excel
with pd.ExcelWriter('multiple.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
    df3.to_excel(writer, sheet_name='Sheet3')
```

### Export Options

```python
# CSV options
df.to_csv('data.csv',
    sep=';',              # separator
    index=False,          # exclude index
    header=True,          # include header
    encoding='utf-8'      # encoding
)

# Excel options
df.to_excel('data.xlsx',
    index=False,
    sheet_name='Data',
    startrow=0,           # start row
    startcol=0            # start column
)
```

---

## 📝 Praktik: Data Pipeline

```python
import pandas as pd
import numpy as np

# 1. LOAD DATA
print("=== LOAD DATA ===")
data = {
    'tanggal': pd.date_range('2024-01-01', periods=100),
    'kategori': np.random.choice(['A', 'B', 'C'], 100),
    'penjualan': np.random.randint(50, 200, 100),
    'biaya': np.random.randint(20, 100, 100),
    'region': np.random.choice(['Jakarta', 'Bandung', 'Surabaya'], 100)
}
df = pd.DataFrame(data)
print(f"Loaded {len(df)} records")

# 2. CLEANING
print("\n=== CLEANING ===")
df['profit'] = df['penjualan'] - df['biaya']
df = df[df['profit'] > 0]  # Remove negative profit
print(f"After cleaning: {len(df)} records")

# 3. AGGREGATION
print("\n=== AGGREGATION ===")
summary = df.groupby('kategori').agg({
    'penjualan': 'sum',
    'biaya': 'sum',
    'profit': ['sum', 'mean']
})
print(summary)

# 4. SORTING
print("\n=== TOP REGIONS ===")
top_region = df.groupby('region')['penjualan'].sum().sort_values(ascending=False)
print(top_region)

# 5. EXPORT
df.to_csv('processed_data.csv', index=False)
summary.to_csv('summary.csv')
print("\n✓ Data exported!")
```

---

## 📝 Ringkasan Halaman Ini

### Data Operations

| Operation | Contoh |
| --------- | ------ |
| Sum/Mean | `df['col'].sum()`, `df.groupby().mean()` |
| Count | `df['col'].count()`, `value_counts()` |
| Unique | `df['col'].unique()`, `nunique()` |
| GroupBy | `df.groupby('col').agg(...)` |
| Transform | `df.groupby().transform()` |
| Join | `pd.merge(df1, df2)` |
| Save | `df.to_csv()`, `df.to_excel()` |

---

## ✏️ Latihan

### Latihan 1: Aggregation

```python
df = pd.DataFrame({
    'kategori': ['A', 'A', 'B', 'B', 'C'],
    'nilai': [10, 20, 15, 25, 30]
})

# 1. Sum per kategori
print(df.groupby('kategori')['nilai'].sum())

# 2. Mean per kategori
print(df.groupby('kategori')['nilai'].mean())

# 3. Min-Max per kategori
print(df.groupby('kategori')['nilai'].agg(['min', 'max']))
```

### Latihan 2: Merge

```python
df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'nama': ['Andi', 'Budi', 'Citra']
})

df2 = pd.DataFrame({
    'id': [1, 2, 3],
    'nilai': [85, 90, 78]
})

# Merge inner join
result = pd.merge(df1, df2, on='id')
print(result)
```

### Latihan 3: Save

```python
df = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra'],
    'nilai': [85, 90, 78]
})

# Save ke CSV
df.to_csv('hasil.csv', index=False)

# Save ke Excel
df.to_excel('hasil.xlsx', index=False)

# Save ke JSON
df.to_json('hasil.json', orient='records')
```

---

## 🔗 Referensi

- [Pandas Groupby](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [Pandas Merging](https://pandas.pydata.org/docs/user_guide/merging.html)
- [Pandas IO Tools](https://pandas.pydata.org/docs/user_guide/io.html)
