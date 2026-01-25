---
title: NumPy Advanced
description: Advanced NumPy operations untuk data manipulation
sidebar:
  order: 3
---

## 🚀 NumPy Advanced Operations

### Boolean Indexing

Boolean indexing menggunakan kondisi untuk filter array:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create boolean mask
mask = arr > 5
print(mask)  # [False False False False False  True  True  True  True  True]

# Filter dengan mask
print(arr[mask])  # [6 7 8 9 10]

# Kombinasi dalam satu line
print(arr[arr > 5])      # [6 7 8 9 10]
print(arr[arr % 2 == 0]) # [2 4 6 8 10]
print(arr[arr < 5])      # [1 2 3 4]

# Kombinasi multiple conditions
print(arr[(arr > 3) & (arr < 8)])   # [4 5 6 7]
print(arr[(arr < 3) | (arr > 8)])   # [1 2 9 10]
print(arr[~(arr > 5)])              # [1 2 3 4 5]

# Modifikasi dengan boolean indexing
arr[arr > 5] = 0  # Set semua elemen > 5 menjadi 0
print(arr)  # [1 2 3 4 5 0 0 0 0 0]
```

### Fancy Indexing

Index array menggunakan array dari indices:

```python
arr = np.array([10, 20, 30, 40, 50])

# Ambil elemen tertentu
indices = np.array([0, 2, 4])
print(arr[indices])  # [10 30 50]

# 2D array fancy indexing
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Ambil elemen spesifik
rows = np.array([0, 1, 2])
cols = np.array([0, 1, 2])
print(matrix[rows, cols])  # [1 5 9] - diagonal!
```

### Reshape dan Flatten

```python
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# Reshape - ubah dimensi (total elemen tetap)
matrix = arr.reshape(3, 4)
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 3D reshape
tensor = arr.reshape(2, 2, 3)
print(tensor.shape)  # (2, 2, 3)

# Flatten - jadikan 1D
flat = matrix.flatten()
print(flat)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# Ravel - seperti flatten tapi lebih efficient
flat2 = matrix.ravel()
print(flat2)

# Transpose
print(matrix.T)
# [[ 0  4  8]
#  [ 1  5  9]
#  [ 2  6 10]
#  [ 3  7 11]]
```

---

## 🔀 Concatenation dan Stacking

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate (hanya 1D)
result = np.concatenate([arr1, arr2])
print(result)  # [1 2 3 4 5 6]

# 2D concatenation
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Axis 0 - vertical (stack rows)
result_v = np.concatenate([matrix1, matrix2], axis=0)
print(result_v)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Axis 1 - horizontal (stack columns)
result_h = np.concatenate([matrix1, matrix2], axis=1)
print(result_h)
# [[1 2 5 6]
#  [3 4 7 8]]

# Stack
stacked = np.stack([arr1, arr2])  # buat 2D dari dua 1D
print(stacked)
# [[1 2 3]
#  [4 5 6]]

# Hstack dan vstack
hstack_result = np.hstack([arr1, arr2])  # horizontal
vstack_result = np.vstack([arr1, arr2])  # vertical
```

---

## 🔄 Broadcasting

Broadcasting adalah mekanisme NumPy untuk operasi array dengan shape berbeda:

```python
# Scalar broadcasting
arr = np.array([1, 2, 3])
result = arr + 10  # [11 12 13]

# 1D + 2D
a = np.array([1, 2, 3])           # shape (3,)
b = np.array([[1], [2], [3]])     # shape (3, 1)
result = a + b
print(result)
# [[2 3 4]
#  [3 4 5]
#  [4 5 6]]

# Broadcasting rules:
# 1. Arrays dibandingkan dari trailing dimensions
# 2. Dimensi compatible jika sama atau salah satunya 1
# 3. Dimensi yang lebih sedikit di-pad dengan 1 di depan
```

---

## 📊 Operasi Statistik Lanjut

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Percentile dan quantile
print(np.percentile(arr, 25))  # 3.25 (Q1)
print(np.percentile(arr, 50))  # 5.5 (median)
print(np.percentile(arr, 75))  # 7.75 (Q3)

# Cumulative operations
print(np.cumsum(arr))      # [1 3 6 10 15 21 28 36 45 55]
print(np.cumprod(arr))     # [1 2 6 24 120 ...]

# Sorting
unsorted = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(np.sort(unsorted))   # [1 1 2 3 4 5 6 9]
print(np.argsort(unsorted))  # [1 3 6 0 2 4 7 5] - indices jika di-sort

# Unique values
print(np.unique(unsorted))  # [1 2 3 4 5 6 9]

# Histogram (count bins)
hist, bins = np.histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], bins=4)
print(f"Histogram: {hist}")  # [1 2 3 4]
print(f"Bins: {bins}")
```

---

## 🧮 Linear Algebra Operations

```python
import numpy as np

# Dot product (inner product)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 32 (1*4 + 2*5 + 3*6)

# Matrix multiplication
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print(np.dot(matrix1, matrix2))
# [[19 22]
#  [43 50]]

# Alternative: @ operator (Python 3.5+)
result = matrix1 @ matrix2

# Determinant
print(np.linalg.det(matrix1))

# Inverse
print(np.linalg.inv(matrix1))

# Transpose
print(matrix1.T)

# Eigenvalues dan eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix1)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors: {eigenvectors}")

# Norm
print(np.linalg.norm(a))  # magnitude of vector

# Solve linear equation Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)
print(x)  # [2 3]
```

---

## 🎲 Random Number Generation

```python
import numpy as np

# Set seed untuk reproducibility
np.random.seed(42)

# Uniform random (0 to 1)
print(np.random.rand(5))  # [0.37 0.95 0.73 0.60 0.16]
print(np.random.rand(3, 3))  # 3x3 random matrix

# Normal distribution (Gaussian)
print(np.random.randn(5))  # mean=0, std=1

# Random integers
print(np.random.randint(0, 10, 5))  # 5 random integers dari 0-9

# Random choice dari array
arr = np.array(['a', 'b', 'c', 'd'])
print(np.random.choice(arr, 3))  # ['b' 'd' 'a']

# Shuffle
arr = np.arange(10)
np.random.shuffle(arr)
print(arr)  # shuffled

# Different distributions
print(np.random.binomial(10, 0.5, 5))  # binomial distribution
print(np.random.exponential(2, 5))     # exponential distribution
print(np.random.poisson(3, 5))         # poisson distribution
```

---

## 📝 Praktik: Data Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate test scores untuk 1000 students
np.random.seed(42)
scores = np.random.normal(loc=75, scale=12, size=1000)

# Clip ke range 0-100
scores = np.clip(scores, 0, 100)

print(f"Mean: {scores.mean():.2f}")
print(f"Std: {scores.std():.2f}")
print(f"Min: {scores.min():.2f}")
print(f"Max: {scores.max():.2f}")

# Percentiles
print(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95]:
    print(f"{p}th: {np.percentile(scores, p):.2f}")

# Count dalam range tertentu
passing = (scores >= 60).sum()
high_pass = (scores >= 80).sum()
print(f"\nPassing (>=60): {passing}/1000")
print(f"High pass (>=80): {high_pass}/1000")

# Histogram
plt.hist(scores, bins=30, edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Distribution of Test Scores')
plt.show()
```

---

## 📝 Ringkasan Halaman Ini

### Advanced Operations

| Operation | Contoh |
| --------- | ------ |
| Boolean Indexing | `arr[arr > 5]` |
| Reshape | `arr.reshape(3, 4)` |
| Concatenate | `np.concatenate([a, b])` |
| Broadcasting | `arr + scalar` |
| Statistics | `np.percentile()`, `np.unique()` |
| Linear Algebra | `np.dot()`, `np.linalg.inv()` |
| Random | `np.random.randn()`, `np.random.choice()` |

---

## ✏️ Latihan

### Latihan 1: Boolean Indexing

```python
arr = np.array([10, 25, 35, 42, 18, 50, 65, 30, 15])

# 1. Filter nilai > 30
print(arr[arr > 30])

# 2. Filter nilai antara 20-50
print(arr[(arr >= 20) & (arr <= 50)])

# 3. Ganti nilai > 50 dengan 100
arr_copy = arr.copy()
arr_copy[arr_copy > 50] = 100
print(arr_copy)
```

### Latihan 2: Reshape dan Aggregate

```python
arr = np.arange(12)

# 1. Reshape menjadi 3x4
matrix = arr.reshape(3, 4)
print(matrix)

# 2. Sum per row
print(matrix.sum(axis=1))

# 3. Mean per column
print(matrix.mean(axis=0))

# 4. Transpose
print(matrix.T)
```

### Latihan 3: Random Data Simulation

```python
np.random.seed(42)

# 1. Generate 100 random scores (mean=75, std=10)
scores = np.random.normal(75, 10, 100)
scores = np.clip(scores, 0, 100)

# 2. Find percentage >= 80
high_scores = (scores >= 80).sum()
percentage = (high_scores / len(scores)) * 100
print(f"Percentage >= 80: {percentage:.1f}%")

# 3. Find quartiles
print(f"Q1: {np.percentile(scores, 25):.2f}")
print(f"Q2: {np.percentile(scores, 50):.2f}")
print(f"Q3: {np.percentile(scores, 75):.2f}")
```

---

## 🔗 Referensi

- [NumPy Advanced Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy Random](https://numpy.org/doc/stable/reference/random/index.html)
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
