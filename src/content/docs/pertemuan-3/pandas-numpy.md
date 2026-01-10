---
title: Manipulasi Data dengan Pandas & NumPy
description: Menguasai library utama untuk pengolahan data tabular dan komputasi numerik
sidebar:
  order: 1
---

## 🎯 Tujuan Pembelajaran

![Data Analysis](https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=400&fit=crop)
_Ilustrasi: Pandas dan NumPy adalah fondasi pengolahan data di Python_

Setelah mempelajari materi ini, kamu akan mampu:

- ✅ Memahami struktur data NumPy Array
- ✅ Membuat dan memanipulasi DataFrame Pandas
- ✅ Melakukan operasi dasar pada data tabular
- ✅ Mengakses, memfilter, dan mengubah data

---

## 📦 Instalasi Library

:::tip[Untuk Pengguna Google Colab]
Jika menggunakan **Google Colab**, library Pandas dan NumPy sudah terinstall otomatis! Langsung lanjut ke bagian import saja.

Buka Google Colab di: [colab.research.google.com](https://colab.research.google.com)
:::

Jika menggunakan laptop/PC lokal:

```bash
pip install pandas numpy
```

Import library:

```python
import pandas as pd
import numpy as np
```

---

## 🔢 NumPy: Numerical Python

### Apa itu NumPy?

NumPy adalah library fundamental untuk komputasi numerik di Python. Keunggulannya:

- ⚡ Lebih cepat dari list Python biasa
- 📊 Operasi matematika yang efisien
- 🧮 Dukungan untuk array multidimensi

### Membuat Array

Array di NumPy adalah koleksi elemen yang homogen (satu tipe data saja), sangat cepat untuk komputasi matematika.

```python
import numpy as np

# Dari list - convert Python list menjadi numpy array (lebih cepat operasinya)
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # [1 2 3 4 5]

# Array 2D (matrix) - array dengan 2 dimensi (baris dan kolom)
arr2d = np.array([
    [1, 2, 3],   # baris pertama: 3 elemen
    [4, 5, 6],   # baris kedua: 3 elemen
    [7, 8, 9]    # baris ketiga: 3 elemen
])
print(arr2d)  # matrix 3x3

# Array dengan fungsi khusus - membuat array dengan pattern tertentu
zeros = np.zeros(5)              # [0. 0. 0. 0. 0.] - 5 elemen semua 0
ones = np.ones((3, 3))           # Matrix 3x3 semua 1
range_arr = np.arange(0, 10, 2)  # [0 2 4 6 8] - dari 0 sampai 10 step 2
linspace = np.linspace(0, 1, 5)  # [0. 0.25 0.5 0.75 1.] - 5 elemen spread dari 0 ke 1

# Penjelasan:
# np.zeros(n) - buat array dengan n elemen, isi 0
# np.ones(shape) - buat array dengan shape tertentu, isi 1
# np.arange(start, end, step) - seperti range() tapi untuk array
# np.linspace(start, end, num) - buat num elemen spread equal dari start ke end
```

### Atribut Array

Atribut adalah informasi tentang array - berapa dimensi, berapa elemen, tipe data apa, dst.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # array 2x3

# Informasi tentang array
print(arr.shape)   # (2, 3) - 2 baris, 3 kolom
print(arr.ndim)    # 2 - dimensi (1D=1, 2D=2, 3D=3, dst)
print(arr.size)    # 6 - total jumlah elemen (2*3=6)
print(arr.dtype)   # int64 - tipe data (int, float, string, dll)

# Penjelasan:
# shape - tuple berisi jumlah elemen per dimensi
# ndim - jumlah dimensi (rank) array
# size - total elemen dalam array
# dtype - tipe data dari setiap elemen
```

### Operasi Matematika

NumPy bisa melakukan operasi matematika pada semua elemen array sekaligus (element-wise), ini jauh lebih cepat daripada loop manual.

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Operasi element-wise (dilakukan ke setiap elemen)
print(a + b)    # [11 22 33 44 55] - penjumlahan setiap elemen
print(a * b)    # [10 40 90 160 250] - perkalian setiap elemen
print(a ** 2)   # [1 4 9 16 25] - kuadrat setiap elemen
print(np.sqrt(a))  # [1. 1.41 1.73 2. 2.24] - akar setiap elemen

# Operasi statistik (menghasilkan 1 nilai)
print(a.sum())    # 15 - jumlah semua elemen
print(a.mean())   # 3.0 - rata-rata
print(a.std())    # 1.41 - standard deviation
print(a.min())    # 1 - elemen terkecil
print(a.max())    # 5 - elemen terbesar

# Penjelasan:
# Element-wise operation dilakukan parallel ke semua elemen, sangat cepat!
# .sum(), .mean(), .std() dll adalah aggregation functions (1 nilai output)
# Ini jauh lebih cepat daripada loop manual di Python
```

### Indexing dan Slicing

```python
arr = np.array([10, 20, 30, 40, 50])

# Indexing
print(arr[0])     # 10
print(arr[-1])    # 50

# Slicing
print(arr[1:4])   # [20 30 40]
print(arr[:3])    # [10 20 30]
print(arr[::2])   # [10 30 50]

# 2D Array
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix[0, 0])     # 1 (baris 0, kolom 0)
print(matrix[1, :])     # [4 5 6] (baris 1, semua kolom)
print(matrix[:, 2])     # [3 6 9] (semua baris, kolom 2)
print(matrix[0:2, 1:3]) # [[2 3] [5 6]]
```

### Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Filter dengan kondisi
print(arr > 5)           # [False False ... True True True]
print(arr[arr > 5])      # [6 7 8 9 10]
print(arr[arr % 2 == 0]) # [2 4 6 8 10]

# Kombinasi kondisi
print(arr[(arr > 3) & (arr < 8)])  # [4 5 6 7]
```

### Reshape Array

```python
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# Reshape menjadi 3x4
matrix = arr.reshape(3, 4)
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Flatten kembali
flat = matrix.flatten()
print(flat)  # [ 0  1  2 ... 11]
```

---

## 🐼 Pandas: Data Analysis Library

### Apa itu Pandas?

Pandas adalah library untuk analisis dan manipulasi data. Dua struktur data utama:

- **Series**: Array 1 dimensi dengan label
- **DataFrame**: Tabel 2 dimensi (seperti spreadsheet/tabel database)

### Series

Series adalah koleksi data 1 dimensi dengan label untuk setiap index (seperti kolom dalam spreadsheet).

```python
import pandas as pd

# Membuat Series dari list
nilai = pd.Series([85, 90, 78, 92, 88])
print(nilai)
# 0    85  <- index 0, nilai 85
# 1    90  <- index 1, nilai 90
# 2    78
# 3    92
# 4    88
# dtype: int64

# Dengan index custom (label sendiri)
nilai = pd.Series(
    [85, 90, 78, 92, 88],
    index=['Matematika', 'Fisika', 'Kimia', 'Biologi', 'Bahasa']
)
print(nilai['Matematika'])  # 85 - akses dengan label
print(nilai.mean())         # 86.6 - rata-rata semua nilai

# Penjelasan:
# Series adalah array dengan index (default: 0, 1, 2, ... atau custom)
# Bisa akses dengan nilai index (`.iloc[0]`) atau label index (`.loc['Matematika']`)
# Series punya built-in statistical methods (.mean(), .sum(), .max(), dll)
```

### DataFrame: Membuat Data

DataFrame adalah tabel 2 dimensi dengan baris dan kolom (seperti Excel atau SQL table). Ini adalah struktur data paling penting di Pandas untuk data analysis.

```python
# Dari dictionary - key menjadi nama kolom
data = {
    'nama': ['Andi', 'Budi', 'Citra', 'Diana', 'Eko'],
    'umur': [20, 21, 19, 22, 20],
    'jurusan': ['SI', 'TI', 'SI', 'TI', 'SI'],
    'ipk': [3.5, 3.7, 3.2, 3.8, 3.4]
}

df = pd.DataFrame(data)  # convert dictionary ke DataFrame
print(df)
#     nama  umur jurusan  ipk
# 0   Andi    20      SI  3.5
# 1   Budi    21      TI  3.7
# 2  Citra    19      SI  3.2
# 3  Diana    22      TI  3.8
# 4    Eko    20      SI  3.4

# Setiap key dalam dictionary menjadi kolom
# Jumlah value di setiap key harus sama (semua 5 elemen)
# Index otomatis 0, 1, 2, 3, 4 (bisa di-custom)

# Penjelasan:
# DataFrame adalah 2D array dengan label untuk baris dan kolom
# Struktur: pd.DataFrame(dictionary) di mana key=column name, value=column data
# Setiap column adalah Series
```

### Membaca File CSV

```python
# Baca dari file CSV
df = pd.read_csv('data.csv')

# Baca dari URL
url = 'https://raw.githubusercontent.com/example/data.csv'
df = pd.read_csv(url)

# Opsi tambahan
df = pd.read_csv('data.csv',
                 sep=';',           # delimiter
                 header=0,          # baris header
                 index_col='id',    # kolom index
                 na_values=['NA', 'null'])  # nilai missing
```

### Eksplorasi Data Awal

```python
# Lihat beberapa baris pertama/terakhir
print(df.head())      # 5 baris pertama
print(df.head(10))    # 10 baris pertama
print(df.tail())      # 5 baris terakhir

# Info dataset
print(df.shape)       # (rows, columns)
print(df.columns)     # nama kolom
print(df.dtypes)      # tipe data tiap kolom
print(df.info())      # ringkasan lengkap

# Statistik deskriptif
print(df.describe())  # count, mean, std, min, 25%, 50%, 75%, max
```

---

## 🔍 Mengakses Data

### Mengakses Kolom

```python
# Satu kolom (Series)
print(df['nama'])
print(df.nama)  # cara alternatif

# Multiple kolom (DataFrame)
print(df[['nama', 'ipk']])
```

### Mengakses Baris dengan loc dan iloc

```python
# loc - berdasarkan label/index name
print(df.loc[0])                    # baris pertama
print(df.loc[0:2])                  # baris 0, 1, 2 (inclusive)
print(df.loc[0, 'nama'])            # nilai spesifik
print(df.loc[0:2, ['nama', 'ipk']]) # subset

# iloc - berdasarkan posisi integer
print(df.iloc[0])                   # baris pertama
print(df.iloc[0:2])                 # baris 0, 1 (exclusive)
print(df.iloc[0, 0])                # baris 0, kolom 0
print(df.iloc[0:3, 0:2])            # slice
```

### Filtering Data

```python
# Filter dengan kondisi
df_tinggi = df[df['ipk'] >= 3.5]
print(df_tinggi)

# Multiple kondisi
df_si_tinggi = df[(df['jurusan'] == 'SI') & (df['ipk'] >= 3.4)]
print(df_si_tinggi)

# Menggunakan isin
df_jurusan = df[df['jurusan'].isin(['SI', 'TI'])]

# Menggunakan query (lebih readable)
df_query = df.query('umur >= 20 and ipk >= 3.5')
print(df_query)
```

---

## ✏️ Memodifikasi Data

### Menambah Kolom

```python
# Kolom baru dari perhitungan
df['nilai_akhir'] = df['ipk'] * 25

# Kolom baru dengan kondisi
df['status'] = df['ipk'].apply(lambda x: 'Lulus' if x >= 3.0 else 'Tidak Lulus')

# Menggunakan np.where
df['kategori'] = np.where(df['ipk'] >= 3.5, 'Cum Laude', 'Reguler')
```

### Mengubah Nilai

```python
# Mengubah nilai tertentu
df.loc[0, 'ipk'] = 3.6

# Mengubah berdasarkan kondisi
df.loc[df['jurusan'] == 'SI', 'jurusan'] = 'Sistem Informasi'

# Replace values
df['jurusan'] = df['jurusan'].replace({'SI': 'Sistem Informasi', 'TI': 'Teknik Informatika'})
```

### Menghapus Data

```python
# Hapus kolom
df_new = df.drop('nilai_akhir', axis=1)
df_new = df.drop(['kolom1', 'kolom2'], axis=1)

# Hapus baris
df_new = df.drop(0, axis=0)  # hapus baris index 0
df_new = df.drop([0, 1, 2], axis=0)  # hapus multiple baris

# Inplace (mengubah df asli)
df.drop('kolom', axis=1, inplace=True)
```

### Rename Kolom

```python
# Rename kolom tertentu
df = df.rename(columns={'nama': 'Nama Lengkap', 'ipk': 'IPK'})

# Rename semua kolom
df.columns = ['nama', 'usia', 'prodi', 'nilai']

# Lowercase semua kolom
df.columns = df.columns.str.lower()
```

---

## 📊 Operasi pada Data

### Sorting

```python
# Sort by satu kolom
df_sorted = df.sort_values('ipk')              # ascending
df_sorted = df.sort_values('ipk', ascending=False)  # descending

# Sort by multiple kolom
df_sorted = df.sort_values(['jurusan', 'ipk'], ascending=[True, False])

# Sort by index
df_sorted = df.sort_index()
```

### Agregasi

```python
# Statistik dasar
print(df['ipk'].sum())
print(df['ipk'].mean())
print(df['ipk'].median())
print(df['ipk'].std())
print(df['ipk'].min())
print(df['ipk'].max())
print(df['ipk'].count())

# Value counts
print(df['jurusan'].value_counts())
# SI    3
# TI    2

# Unique values
print(df['jurusan'].unique())      # ['SI', 'TI']
print(df['jurusan'].nunique())     # 2
```

### Group By

```python
# Agregasi per grup
print(df.groupby('jurusan')['ipk'].mean())
# jurusan
# SI    3.37
# TI    3.75

# Multiple agregasi
print(df.groupby('jurusan').agg({
    'ipk': ['mean', 'min', 'max'],
    'umur': 'mean'
}))

# Reset index
result = df.groupby('jurusan')['ipk'].mean().reset_index()
result.columns = ['jurusan', 'rata_rata_ipk']
```

---

## 🔗 Menggabungkan Data

### Concat

```python
# Gabung vertikal (row-wise)
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df_concat = pd.concat([df1, df2], ignore_index=True)

# Gabung horizontal (column-wise)
df_concat = pd.concat([df1, df2], axis=1)
```

### Merge (Join)

```python
# Data mahasiswa
mahasiswa = pd.DataFrame({
    'id': [1, 2, 3],
    'nama': ['Andi', 'Budi', 'Citra']
})

# Data nilai
nilai = pd.DataFrame({
    'id': [1, 2, 3],
    'mata_kuliah': ['Statistik', 'Database', 'Algoritma'],
    'nilai': [85, 90, 78]
})

# Inner join (default)
result = pd.merge(mahasiswa, nilai, on='id')

# Left join
result = pd.merge(mahasiswa, nilai, on='id', how='left')

# Right join
result = pd.merge(mahasiswa, nilai, on='id', how='right')

# Outer join
result = pd.merge(mahasiswa, nilai, on='id', how='outer')
```

---

## 💾 Menyimpan Data

```python
# Simpan ke CSV
df.to_csv('output.csv', index=False)

# Simpan ke Excel
df.to_excel('output.xlsx', index=False, sheet_name='Sheet1')

# Simpan ke JSON
df.to_json('output.json', orient='records')
```

---

## 📝 Praktik: Mini Project

### Analisis Data Mahasiswa

```python
import pandas as pd
import numpy as np

# Buat data sample
np.random.seed(42)
data = {
    'nama': [f'Mahasiswa_{i}' for i in range(1, 101)],
    'angkatan': np.random.choice([2022, 2023, 2024], 100),
    'jurusan': np.random.choice(['SI', 'TI', 'MI'], 100),
    'ipk': np.round(np.random.uniform(2.0, 4.0, 100), 2),
    'sks': np.random.randint(100, 145, 100)
}
df = pd.DataFrame(data)

# 1. Lihat data awal
print("=== 5 Data Pertama ===")
print(df.head())

# 2. Statistik deskriptif
print("\n=== Statistik Deskriptif ===")
print(df.describe())

# 3. Rata-rata IPK per jurusan
print("\n=== Rata-rata IPK per Jurusan ===")
print(df.groupby('jurusan')['ipk'].mean())

# 4. Top 10 mahasiswa IPK tertinggi
print("\n=== Top 10 IPK Tertinggi ===")
top10 = df.nlargest(10, 'ipk')[['nama', 'jurusan', 'ipk']]
print(top10)

# 5. Jumlah mahasiswa per angkatan
print("\n=== Jumlah per Angkatan ===")
print(df['angkatan'].value_counts().sort_index())
```

---

## 📝 Ringkasan

| Library | Struktur Data | Kegunaan                 |
| ------- | ------------- | ------------------------ |
| NumPy   | Array         | Komputasi numerik cepat  |
| Pandas  | Series        | Array 1D dengan label    |
| Pandas  | DataFrame     | Tabel 2D (seperti Excel) |

### Operasi Penting

| Operasi   | NumPy                | Pandas                  |
| --------- | -------------------- | ----------------------- |
| Membuat   | `np.array()`         | `pd.DataFrame()`        |
| Akses     | `arr[0]`, `arr[0:5]` | `df.loc[]`, `df.iloc[]` |
| Filter    | `arr[arr > 5]`       | `df[df['col'] > 5]`     |
| Statistik | `arr.mean()`         | `df['col'].mean()`      |

---

## ✏️ Latihan

### Latihan 1: NumPy

1. Buat array 1D berisi angka 1-20
2. Reshape menjadi matrix 4x5
3. Hitung sum setiap baris dan kolom

### Latihan 2: Pandas

1. Buat DataFrame dengan 5 kolom data fiktif
2. Lakukan filtering berdasarkan 2 kondisi
3. Hitung statistik deskriptif per grup

### Latihan 3: Data Real

1. Download dataset CSV dari Kaggle
2. Eksplorasi dengan head(), info(), describe()
3. Temukan 3 insights menarik dari data

---

## ❓ FAQ (Pertanyaan yang Sering Diajukan)

### Q: NumPy vs Python list, apa bedanya?

**A:** NumPy array jauh lebih cepat dan hemat memory:

- **Python list**: Flexible, bisa isi berbagai tipe, tapi lambat untuk operasi numerik
- **NumPy array**: Hanya satu tipe, tapi operasi 10-100x lebih cepat karena optimized untuk komputasi

Untuk data science gunakan NumPy!

### Q: Apa itu element-wise operation?

**A:** Operasi yang dilakukan ke setiap elemen. Contoh:

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
c = a + b  # [1+10, 2+20, 3+30] = [11, 22, 33]
```

Bukan menambah seluruh array, tapi setiap elemen ditambah dengan element corresponding.

### Q: Apa bedanya .loc[] vs .iloc[]?

**A:**

- `.loc[]` - akses berdasarkan **label** (nama) index
- `.iloc[]` - akses berdasarkan **posisi** (integer location)

Contoh:

```python
df.loc[0]     # baris dengan label/index 0
df.iloc[0]    # baris pertama (posisi 0)
```

Keduanya biasanya sama jika index adalah 0, 1, 2,... tapi kalau index adalah custom (nama, tanggal, dll) beda hasilnya!

### Q: Bagaimana filtering data dengan multiple kondisi?

**A:** Gunakan `&` (AND) atau `|` (OR):

```python
# AND - harus keduanya true
df[(df['ipk'] >= 3.5) & (df['jurusan'] == 'SI')]

# OR - minimal satu true
df[(df['ipk'] >= 3.5) | (df['umur'] >= 21)]

# NOT - negasi
df[~(df['jurusan'] == 'SI')]  # yang bukan SI
```

Perhatian: gunakan `&` dan `|`, jangan `and` dan `or` (syntax berbeda!)

### Q: Saya punya file CSV, bagaimana cara baca di Pandas?

**A:** Sangat mudah:

```python
df = pd.read_csv('data.csv')
```

Beberapa opsi common:

```python
df = pd.read_csv('data.csv',
                  sep=';',        # delimiter (default comma)
                  encoding='utf-8',  # encoding
                  na_values=['NA', 'null'])  # nilai apa yang dianggap missing
```

### Q: Bagaimana cara simpan DataFrame ke file?

**A:** Ada beberapa format:

```python
df.to_csv('output.csv', index=False)     # CSV
df.to_excel('output.xlsx', index=False)  # Excel
df.to_json('output.json')                 # JSON
df.to_sql('tabel', con=connection)       # Database
```

`index=False` agar index tidak disimpan (optional).

### Q: Apa bedanya Series vs DataFrame?

**A:**

- **Series**: 1 dimensi (seperti 1 kolom)
- **DataFrame**: 2 dimensi (seperti tabel dengan banyak kolom)

Series adalah "building block" dari DataFrame. Setiap kolom DataFrame adalah Series.

### Q: Groupby itu apa? Kapan pakai?

**A:** Groupby digunakan untuk membagi data ke grup lalu hitung statistik per grup. Contoh:

```python
# Hitung rata-rata IPK per jurusan
df.groupby('jurusan')['ipk'].mean()
# SI    3.37
# TI    3.75
```

Ini lebih cepat daripada loop manual!

---

:::note[Catatan]
Pertemuan ini mengenalkan dua library terpenting untuk data science. Kuasai keduanya dengan praktik berulang-ulang. Jangan langsung hafal, pahami konsepnya dulu, sisanya akan ingat dengan sendirinya.
