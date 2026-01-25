---
title: NumPy Fundamentals
description: Memahami dasar-dasar NumPy untuk komputasi numerik
sidebar:
  order: 2
---

## 🔢 NumPy: Numerical Python

### Apa itu NumPy?

NumPy adalah library fundamental untuk komputasi numerik di Python. Ini adalah fondasi dari hampir semua library data science lainnya.

**Keunggulan NumPy:**
- ⚡ 10-100x lebih cepat dari list Python biasa
- 📊 Operasi matematika yang efisien
- 🧮 Dukungan untuk array multidimensi (matrix, tensor)
- 🔄 Seamless integration dengan library lain (Pandas, Scikit-learn, TensorFlow)

### Instalasi dan Import

```python
# Install (jika belum)
# pip install numpy

# Import
import numpy as np

# Check version
print(np.__version__)
```

---

## 📦 Membuat Array

### Dari List Python

```python
import numpy as np

# 1D array dari list
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)
# Output: [1 2 3 4 5]

# 2D array dari list of lists (matrix)
arr2d = np.array([
    [1, 2, 3],      # baris 0
    [4, 5, 6],      # baris 1
    [7, 8, 9]       # baris 2
])
print(arr2d)
# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# 3D array
arr3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(arr3d.shape)  # (2, 2, 2)
```

### Dengan Built-in Functions

NumPy menyediakan banyak fungsi untuk membuat array dengan pattern tertentu:

```python
# Array dengan nilai 0
zeros = np.zeros(5)
# Output: [0. 0. 0. 0. 0.]

# Array 2D dengan nilai 0
zeros_2d = np.zeros((3, 4))
# Output: [[0. 0. 0. 0.]
#          [0. 0. 0. 0.]
#          [0. 0. 0. 0.]]

# Array dengan nilai 1
ones = np.ones(5)
# Output: [1. 1. 1. 1. 1.]

# Array dengan range nilai
range_arr = np.arange(0, 10, 2)  # start, end, step
# Output: [0 2 4 6 8]

# Array dengan linear spacing (even distribution)
linspace = np.linspace(0, 1, 5)  # start, end, num_points
# Output: [0. 0.25 0.5 0.75 1.]

# Array dengan random values
random = np.random.rand(3, 3)  # 3x3 random array
print(random)

# Array dengan nilai specific
full = np.full(5, 7)  # array dengan 5 elemen, isi semua 7
# Output: [7 7 7 7 7]

# Array dari range dengan float step
arange_float = np.arange(0, 1, 0.1)
# Output: [0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
```

---

## 📊 Atribut Array

Atribut adalah informasi tentang array:

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Shape - dimensi array
print(arr.shape)  # (2, 3) - 2 baris, 3 kolom

# ndim - jumlah dimensi
print(arr.ndim)  # 2 (1D array = ndim 1, 2D = ndim 2, dst)

# size - total elemen
print(arr.size)  # 6 (2 * 3)

# dtype - tipe data
print(arr.dtype)  # int64

# itemsize - bytes per elemen
print(arr.itemsize)  # 8 (untuk int64)

# nbytes - total bytes
print(arr.nbytes)  # 48 (6 * 8)

# Data types yang common
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1, 2, 3], dtype=np.float64)
arr_complex = np.array([1+2j, 3+4j], dtype=np.complex128)
arr_bool = np.array([True, False], dtype=bool)
arr_str = np.array(['hello', 'world'], dtype=str)
```

---

## 🔢 Indexing dan Slicing

### 1D Array

```python
arr = np.array([10, 20, 30, 40, 50])

# Indexing - ambil elemen tertentu
print(arr[0])      # 10 (elemen pertama)
print(arr[2])      # 30 (elemen ketiga)
print(arr[-1])     # 50 (elemen terakhir)
print(arr[-2])     # 40 (elemen kedua dari belakang)

# Slicing - ambil range elemen
print(arr[1:4])    # [20 30 40] (dari index 1 sampai 3, index 4 tidak termasuk)
print(arr[:3])     # [10 20 30] (dari awal sampai index 2)
print(arr[2:])     # [30 40 50] (dari index 2 sampai akhir)
print(arr[::2])    # [10 30 50] (setiap 2 elemen - step 2)
print(arr[::-1])   # [50 40 30 20 10] (reverse - step -1)

# Slicing dengan step negatif
print(arr[4:1:-1]) # [50 40 30] (dari index 4 ke 1, mundur)
```

### 2D Array

```python
matrix = np.array([
    [1, 2, 3],      # row 0
    [4, 5, 6],      # row 1
    [7, 8, 9]       # row 2
])

# Indexing - ambil elemen tertentu
print(matrix[0, 0])     # 1 (baris 0, kolom 0)
print(matrix[1, 2])     # 6 (baris 1, kolom 2)
print(matrix[2, -1])    # 9 (baris 2, kolom terakhir)

# Indexing baris
print(matrix[0])        # [1 2 3] (ambil baris 0 - menjadi 1D)
print(matrix[0, :])     # [1 2 3] (sama, tapi explicit)

# Indexing kolom
print(matrix[:, 0])     # [1 4 7] (ambil kolom 0)
print(matrix[:, 2])     # [3 6 9] (ambil kolom 2)

# Slicing 2D
print(matrix[0:2, 1:3]) # [[2 3]
                        #  [5 6]]
print(matrix[1:, :2])   # [[4 5]
                        #  [7 8]]
```

---

## ✖️ Operasi Matematika (Element-wise)

Element-wise operation berarti operasi dilakukan ke **setiap elemen** dalam array secara bersamaan. Ini sangat cepat karena di-optimize di C level.

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Aritmatika element-wise
print(a + b)      # [11 22 33 44 55]
print(a - b)      # [-9 -18 -27 -36 -45]
print(a * b)      # [10 40 90 160 250]
print(a / b)      # [0.1 0.1 0.1 0.1 0.1]
print(a ** 2)     # [1 4 9 16 25]
print(a % 2)      # [1 0 1 0 1] (modulo)

# Broadcasting - operasi dengan scalar
arr = np.array([1, 2, 3, 4, 5])
print(arr + 10)   # [11 12 13 14 15]
print(arr * 2)    # [2 4 6 8 10]
print(10 - arr)   # [9 8 7 6 5]

# Operasi fungsi mathematical
print(np.sqrt(a))           # [1. 1.41 1.73 2. 2.24]
print(np.exp(a))            # [2.72 7.39 20.09 54.60 148.41]
print(np.log(a))            # [0. 0.69 1.10 1.39 1.61]
print(np.abs(np.array([-1, -2, 3])))  # [1 2 3]
print(np.sin(np.array([0, np.pi/2]))) # [0. 1.]
```

---

## 📈 Operasi Agregasi (Reduction)

Agregasi operations mengurangi dimensi array menjadi satu nilai (atau lebih sedikit dimensi):

```python
arr = np.array([1, 2, 3, 4, 5])

# Basic aggregations
print(arr.sum())        # 15 - jumlah semua elemen
print(arr.mean())       # 3.0 - rata-rata
print(arr.std())        # 1.41 - standard deviation
print(arr.var())        # 2.0 - variance
print(arr.min())        # 1 - minimum
print(arr.max())        # 5 - maksimum
print(arr.argmin())     # 0 - index dari minimum
print(arr.argmax())     # 4 - index dari maksimum

# 2D array aggregations
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Agregasi seluruh array
print(matrix.sum())     # 21 - jumlah semua elemen

# Agregasi per axis
print(matrix.sum(axis=0))  # [5 7 9] - sum per kolom
print(matrix.sum(axis=1))  # [6 15] - sum per baris
print(matrix.mean(axis=0)) # [2.5 3.5 4.5] - mean per kolom
```

---

## 📝 Praktik: Operasi Dasar

```python
# Data nilai ujian 5 mahasiswa, 3 mata kuliah
nilai = np.array([
    [85, 90, 88],  # Mahasiswa 1
    [75, 80, 78],  # Mahasiswa 2
    [92, 95, 90],  # Mahasiswa 3
    [70, 72, 68],  # Mahasiswa 4
    [88, 85, 89]   # Mahasiswa 5
])

print("=== Analisis Nilai ===")
print(f"Shape: {nilai.shape}")  # (5, 3)

# Rata-rata per mahasiswa
print(f"\nRata-rata per mahasiswa:")
print(nilai.mean(axis=1))

# Rata-rata per mata kuliah
print(f"\nRata-rata per mata kuliah:")
print(nilai.mean(axis=0))

# Nilai tertinggi
print(f"\nNilai tertinggi per mahasiswa:")
print(nilai.max(axis=1))

# Statistik keseluruhan
print(f"\nStatistik keseluruhan:")
print(f"Mean: {nilai.mean():.2f}")
print(f"Std: {nilai.std():.2f}")
print(f"Min: {nilai.min()}")
print(f"Max: {nilai.max()}")
```

---

## 📝 Ringkasan Halaman Ini

### Array Basics

| Konsep | Contoh |
| ------ | ------ |
| Create | `np.array([1,2,3])`, `np.zeros(5)`, `np.arange(10)` |
| Shape | `arr.shape` → `(3, 4)` |
| Indexing | `arr[0]`, `arr[1, 2]` |
| Slicing | `arr[1:4]`, `arr[:, 2]` |
| Element-wise | `arr + 5`, `arr * 2`, `np.sqrt(arr)` |
| Aggregation | `arr.sum()`, `arr.mean()`, `arr.max()` |

---

## ✏️ Latihan

### Latihan 1: Array Creation

```python
import numpy as np

# 1. Buat array dari 1-10
arr1 = np.arange(1, 11)
print(f"Array 1-10: {arr1}")

# 2. Buat array dari 0-20 dengan step 2
arr2 = np.arange(0, 21, 2)
print(f"Array 0-20 step 2: {arr2}")

# 3. Buat array 5 nilai yang spread equal dari 0-1
arr3 = np.linspace(0, 1, 5)
print(f"Linspace 0-1, 5 points: {arr3}")

# 4. Buat 3x3 matrix dengan nilai 1
matrix = np.ones((3, 3))
print(f"3x3 ones:\n{matrix}")
```

### Latihan 2: Indexing dan Slicing

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

# 1. Ambil elemen pertama, ketiga, terakhir
print(f"Pertama: {arr[0]}")
print(f"Ketiga: {arr[2]}")
print(f"Terakhir: {arr[-1]}")

# 2. Ambil elemen dari index 2-5
print(f"Index 2-5: {arr[2:6]}")

# 3. Ambil elemen with step 2
print(f"Step 2: {arr[::2]}")

# 4. Reverse array
print(f"Reversed: {arr[::-1]}")
```

### Latihan 3: Operasi Matematika

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# 1. Hitung a + b
result1 = a + b
print(f"a + b: {result1}")

# 2. Hitung a * b
result2 = a * b
print(f"a * b: {result2}")

# 3. Hitung a / b
result3 = a / b
print(f"a / b: {result3}")

# 4. Hitung sqrt(a)
result4 = np.sqrt(a)
print(f"sqrt(a): {result4}")

# 5. Hitung sum, mean, max
print(f"\nAgregasi:")
print(f"Sum: {a.sum()}")
print(f"Mean: {a.mean()}")
print(f"Max: {a.max()}")
```

---

## ❓ FAQ

### Q: Kenapa NumPy array lebih cepat dari Python list?

**A:** Karena:
1. NumPy array homogeny (satu tipe), tidak perlu check tipe setiap operasi
2. Data tersimpan contiguous (bersebelahan) di memory
3. Operasi dilakukan di C level (bukan Python interpreter)

Result: 10-100x lebih cepat!

### Q: Apa itu broadcasting?

**A:** Kemampuan NumPy untuk operasi array dengan ukuran berbeda:

```python
arr = np.array([1, 2, 3])    # shape (3,)
scalar = 10                   # scalar
result = arr + scalar         # [11 12 13]
```

NumPy automatically "broadcast" scalar ke shape array yang match.

### Q: Berapa dimensi array yang bisa dibuat?

**A:** Unlimited! Bisa 1D, 2D, 3D, 4D, dst. Tapi untuk data science biasanya:
- 1D: Series (satu variabel)
- 2D: DataFrame (tabel dengan rows & columns)
- 3D+: Image, video, tensor (advanced)

---

## 🔗 Referensi

- [NumPy Documentation](https://numpy.org/doc/)
- [NumPy Tutorial - Official](https://numpy.org/doc/stable/user/basics.html)
- [NumPy Cheat Sheet](https://datacamp.com/cheat-sheets/numpy-cheat-sheet)
