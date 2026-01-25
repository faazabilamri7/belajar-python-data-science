---
title: DataFrame Manipulation
description: Mengakses, memodifikasi, dan mentransformasi DataFrame
sidebar:
  order: 6
---

## ✏️ Memodifikasi DataFrame

### Menambah Kolom

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra'],
    'nilai_ujian': [85, 90, 78],
    'nilai_tugas': [90, 85, 88]
})

# Tambah kolom baru (dari perhitungan)
df['rata_rata'] = (df['nilai_ujian'] + df['nilai_tugas']) / 2
print(df)
#    nama  nilai_ujian  nilai_tugas  rata_rata
# 0  Andi           85           90       87.5
# 1  Budi           90           85       87.5
# 2 Citra           78           88       83.0

# Tambah dengan scalar value (sama di semua baris)
df['tahun'] = 2024
print(df)

# Tambah dengan Series (harus punya index yang sama)
bonus = pd.Series([5, 10, 5], index=[0, 1, 2])
df['bonus_nilai'] = bonus
print(df)

# Tambah dengan kondisi (np.where atau apply)
df['grade'] = np.where(df['rata_rata'] >= 80, 'Lulus', 'Tidak Lulus')
print(df)
```

### Mengubah Nilai

```python
# Ubah nilai single cell
df.loc[0, 'nama'] = 'Andika'

# Ubah nilai dengan kondisi
df.loc[df['rata_rata'] < 80, 'grade'] = 'Belum Lulus'

# Ubah seluruh kolom
df['tahun'] = 2025

# Replace values
df['nama'] = df['nama'].replace({
    'Andi': 'Andi S',
    'Budi': 'Budi T'
})

# Replace multiple values
df.replace(to_replace={'Lulus': 'L', 'Tidak Lulus': 'TL'})
```

### Menghapus Kolom

```python
# Drop satu kolom
df_new = df.drop('tahun', axis=1)

# Drop multiple kolom
df_new = df.drop(['tahun', 'bonus_nilai'], axis=1)

# Drop inplace (ubah df original, tidak return)
df.drop('tahun', axis=1, inplace=True)

# Drop baris (axis=0)
df_new = df.drop(0, axis=0)  # hapus baris index 0
df_new = df.drop([0, 2], axis=0)  # hapus baris 0 dan 2
```

### Rename Kolom dan Index

```python
# Rename spesifik kolom
df = df.rename(columns={
    'nilai_ujian': 'ujian',
    'nilai_tugas': 'tugas'
})

# Rename semua kolom
df.columns = ['Nama', 'Ujian', 'Tugas', 'Rata-rata', 'Grade']

# Rename lowercase
df.columns = df.columns.str.lower()

# Rename dengan replace spasi
df.columns = df.columns.str.replace(' ', '_')

# Rename index
df = df.rename(index={0: 'siswa_1', 1: 'siswa_2'})

# Set index column
df = df.set_index('nama')  # kolom 'nama' jadi index
df = df.reset_index()      # index jadi kolom
```

---

## 🔍 Filtering dan Sorting

### Filter Rows

```python
df = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra', 'Diana', 'Eko'],
    'umur': [20, 21, 19, 22, 20],
    'jurusan': ['SI', 'TI', 'SI', 'TI', 'SI'],
    'ipk': [3.5, 3.7, 3.2, 3.8, 3.4]
})

# Single condition
print(df[df['umur'] > 20])

# Multiple conditions (gunakan & | ~ untuk AND OR NOT)
print(df[(df['umur'] > 19) & (df['ipk'] >= 3.5)])
print(df[(df['jurusan'] == 'SI') | (df['ipk'] >= 3.7)])
print(df[~(df['jurusan'] == 'TI')])  # NOT

# isin() - check apakah value ada dalam list
print(df[df['jurusan'].isin(['SI', 'MI'])])

# str methods untuk string filtering
df_str = pd.DataFrame({
    'email': ['andi@gmail.com', 'budi@yahoo.com', 'citra@gmail.com']
})
print(df_str[df_str['email'].str.contains('gmail')])
print(df_str[df_str['email'].str.endswith('.com')])

# query() method (lebih readable)
print(df.query('umur > 20'))
print(df.query('umur > 20 and ipk >= 3.5'))
print(df.query('jurusan == "SI" or ipk >= 3.7'))

# between() - range check
print(df[df['umur'].between(20, 21)])

# isnull() / notna()
print(df[df['nama'].notna()])
```

### Sorting

```python
# Sort by satu kolom
df_sorted = df.sort_values('ipk')              # ascending (naik)
df_sorted = df.sort_values('ipk', ascending=False)  # descending (turun)

# Sort by multiple kolom
df_sorted = df.sort_values(['jurusan', 'ipk'], ascending=[True, False])
# Sorted by jurusan (ascending), lalu by ipk (descending)

# Sort by index
df_sorted = df.sort_index()

# Sort inplace
df.sort_values('ipk', inplace=True)

# Largest N rows (tanpa sort semua)
print(df.nlargest(3, 'ipk'))      # 3 baris dengan ipk tertinggi
print(df.nsmallest(2, 'umur'))    # 2 baris dengan umur terendah
```

---

## 🧮 Transformasi Data

### Apply Function

```python
df = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra'],
    'nilai': [85, 90, 78]
})

# Apply to column (Series)
def grade(score):
    if score >= 80:
        return 'A'
    elif score >= 70:
        return 'B'
    else:
        return 'C'

df['grade'] = df['nilai'].apply(grade)

# Lambda function (anonymous)
df['nilai_x2'] = df['nilai'].apply(lambda x: x * 2)
df['nilai_plus_10'] = df['nilai'].apply(lambda x: x + 10)

# Apply to row (axis=1)
df['perubahan'] = df.apply(lambda row: f"{row['nama']}: {row['nilai']}", axis=1)

# Multiple outputs per row
def split_name(name):
    return pd.Series(name.split(' '))

df[['first', 'last']] = df['nama'].apply(lambda x: pd.Series(x.split()))
```

### Map Function

```python
# Map specific values (dictionary)
mapping = {'SI': 'Sistem Informasi', 'TI': 'Teknik Informatika'}
df['jurusan_full'] = df['jurusan'].map(mapping)

# Map with function
df['kategori'] = df['nilai'].map(lambda x: 'Tinggi' if x >= 85 else 'Rendah')
```

### String Operations

```python
df = pd.DataFrame({
    'email': ['ANDI@GMAIL.COM', 'budi@yahoo.com', 'Citra@Gmail.Com'],
    'username': ['andi_2024', 'budi-2024', 'citra.2024']
})

# String methods (via .str accessor)
df['email_lower'] = df['email'].str.lower()
df['email_upper'] = df['email'].str.upper()
df['email_len'] = df['email'].str.len()

# Extract part of string
df['domain'] = df['email'].str.extract(r'@(.+)', expand=False)

# Check/filter
print(df[df['email'].str.contains('gmail')])
print(df[df['email'].str.startswith('A')])
print(df[df['email'].str.endswith('.com')])

# Replace
df['username'] = df['username'].str.replace('_', '-')
df['username'] = df['username'].str.replace(r'\d', 'X', regex=True)
```

---

## 📊 Agregasi dan GroupBy

### GroupBy

GroupBy digunakan untuk membagi data ke grup, lalu hitung statistik per grup.

```python
df = pd.DataFrame({
    'jurusan': ['SI', 'SI', 'TI', 'TI', 'SI'],
    'nama': ['Andi', 'Budi', 'Citra', 'Diana', 'Eko'],
    'ipk': [3.5, 3.7, 3.2, 3.8, 3.4],
    'umur': [20, 21, 19, 22, 20]
})

# Group by satu kolom
print(df.groupby('jurusan')['ipk'].mean())
# jurusan
# SI    3.53
# TI    3.50

# Multiple aggregations
print(df.groupby('jurusan').agg({
    'ipk': ['mean', 'min', 'max'],
    'umur': 'mean'
}))

# Named aggregation
print(df.groupby('jurusan').agg(
    ipk_rata_rata=('ipk', 'mean'),
    ipk_tertinggi=('ipk', 'max'),
    umur_rata_rata=('umur', 'mean')
))

# Multiple group by columns
print(df.groupby(['jurusan', 'umur'])['ipk'].mean())

# Get group
group_si = df[df['jurusan'] == 'SI']

# Iterate groups
for jurusan, group in df.groupby('jurusan'):
    print(f"\nJurusan: {jurusan}")
    print(group)

# Reset index (convert group to column)
result = df.groupby('jurusan')['ipk'].mean().reset_index()
result.columns = ['jurusan', 'ipk_rata_rata']
print(result)
```

### Pivot Table

```python
df = pd.DataFrame({
    'tanggal': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    'produk': ['A', 'B', 'A', 'B'],
    'penjualan': [100, 150, 120, 200]
})

# Pivot table
pivot = df.pivot_table(
    values='penjualan',      # nilai yang di-aggregate
    index='tanggal',         # baris
    columns='produk',        # kolom
    aggfunc='sum'            # aggregation function
)
print(pivot)
#           A    B
# tanggal
# 2024-01-01  100  150
# 2024-01-02  120  200
```

---

## 🔗 Merge dan Concat

### Concat (Gabung vertikal/horizontal)

```python
df1 = pd.DataFrame({
    'nama': ['Andi', 'Budi'],
    'nilai': [85, 90]
})

df2 = pd.DataFrame({
    'nama': ['Citra', 'Diana'],
    'nilai': [78, 92]
})

# Concat vertikal (row-wise)
df_vertical = pd.concat([df1, df2], ignore_index=True)
print(df_vertical)
#    nama  nilai
# 0  Andi     85
# 1  Budi     90
# 2  Citra    78
# 3  Diana    92

# Concat horizontal (column-wise)
df3 = pd.DataFrame({
    'bonus': [5, 10]
})
df_horizontal = pd.concat([df1, df3], axis=1)
print(df_horizontal)
#    nama  nilai  bonus
# 0  Andi     85      5
# 1  Budi     90     10
```

### Merge (Join - seperti SQL)

```python
# Mahasiswa
mahasiswa = pd.DataFrame({
    'id': [1, 2, 3],
    'nama': ['Andi', 'Budi', 'Citra']
})

# Nilai
nilai = pd.DataFrame({
    'id': [1, 2, 3],
    'mata_kuliah': ['Statistik', 'Database', 'Algoritma'],
    'skor': [85, 90, 78]
})

# Inner join (default - hanya data yang ada di keduanya)
result = pd.merge(mahasiswa, nilai, on='id')
print(result)
#   id  nama      mata_kuliah  skor
# 0  1  Andi      Statistik       85
# 1  2  Budi      Database        90
# 2  3  Citra     Algoritma       78

# Left join (semua dari kiri)
result = pd.merge(mahasiswa, nilai, on='id', how='left')

# Right join (semua dari kanan)
result = pd.merge(mahasiswa, nilai, on='id', how='right')

# Outer join (semua dari keduanya)
result = pd.merge(mahasiswa, nilai, on='id', how='outer')

# Join with different key names
# pd.merge(df1, df2, left_on='id1', right_on='id2')
```

---

## 📝 Praktik: Data Processing

```python
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
data = {
    'nama': ['Andi', 'Budi', 'Citra', 'Diana', 'Eko'],
    'jurusan': ['SI', 'TI', 'SI', 'TI', 'SI'],
    'nilai_ujian': np.random.randint(60, 100, 5),
    'nilai_tugas': np.random.randint(70, 100, 5)
}

df = pd.DataFrame(data)

# 1. Tambah kolom rata-rata
df['rata_rata'] = (df['nilai_ujian'] + df['nilai_tugas']) / 2

# 2. Tambah kolom grade
df['grade'] = df['rata_rata'].apply(
    lambda x: 'A' if x >= 80 else 'B' if x >= 70 else 'C'
)

# 3. Filter siswa dengan grade A
df_grade_a = df[df['grade'] == 'A']
print("Siswa dengan grade A:")
print(df_grade_a)

# 4. Rata-rata per jurusan
print("\nRata-rata nilai per jurusan:")
print(df.groupby('jurusan')['rata_rata'].mean())

# 5. Sort by rata-rata (descending)
df_sorted = df.sort_values('rata_rata', ascending=False)
print("\nSorted by rata-rata (tertinggi):")
print(df_sorted[['nama', 'rata_rata', 'grade']])
```

---

## 📝 Ringkasan Halaman Ini

### DataFrame Manipulation

| Operation | Contoh |
| --------- | ------ |
| Add Column | `df['col'] = values` |
| Modify | `df.loc[0, 'col'] = value` |
| Delete | `df.drop('col', axis=1)` |
| Filter | `df[df['col'] > 5]` |
| Sort | `df.sort_values('col')` |
| Apply | `df['col'].apply(func)` |
| GroupBy | `df.groupby('col').sum()` |
| Merge | `pd.merge(df1, df2, on='id')` |

---

## ✏️ Latihan

### Latihan 1: Add & Modify Columns

```python
df = pd.DataFrame({
    'nama': ['Andi', 'Budi', 'Citra'],
    'fisika': [85, 90, 78],
    'kimia': [88, 85, 82]
})

# 1. Add rata-rata
df['rata_rata'] = (df['fisika'] + df['kimia']) / 2

# 2. Add status pass/fail (>= 75)
df['status'] = df['rata_rata'].apply(
    lambda x: 'Lulus' if x >= 75 else 'Tidak Lulus'
)

print(df)
```

### Latihan 2: Filter & Sort

```python
# 1. Filter fisika > 80
print(df[df['fisika'] > 80])

# 2. Sort by rata-rata (descending)
print(df.sort_values('rata-rata', ascending=False))
```

### Latihan 3: GroupBy

```python
df = pd.DataFrame({
    'jurusan': ['SI', 'SI', 'TI', 'TI'],
    'nilai': [85, 90, 78, 92]
})

# 1. Mean per jurusan
print(df.groupby('jurusan')['nilai'].mean())

# 2. Count per jurusan
print(df.groupby('jurusan')['nilai'].count())
```

---

## 🔗 Referensi

- [Pandas User Guide - Manipulation](https://pandas.pydata.org/docs/user_guide/merging.html)
- [Pandas GroupBy](https://pandas.pydata.org/docs/reference/groupby.html)
