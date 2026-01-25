---
title: Functions & Advanced Python
description: Mendefinisikan functions, lambda, dan list comprehension
sidebar:
  order: 6
---

## 🔧 Functions

### Membuat Function

Function adalah blok code yang bisa digunakan ulang berkali-kali.

```python
# Fungsi tanpa parameter
def sapa():
    """Fungsi untuk memberi salam"""
    print("Halo, selamat datang!")

sapa()  # Halo, selamat datang!

# Fungsi dengan parameter
def sapa_nama(nama):
    print(f"Halo, {nama}!")

sapa_nama("Faaza")  # Halo, Faaza!

# Fungsi dengan return value
def tambah(a, b):
    return a + b

hasil = tambah(5, 3)
print(hasil)  # 8

# Fungsi dengan multiple return values
def hitung_statistik(angka):
    """Return min, max, dan sum"""
    return min(angka), max(angka), sum(angka)

minimum, maksimum, total = hitung_statistik([1, 2, 3, 4, 5])
print(f"Min: {minimum}, Max: {maksimum}, Sum: {total}")
```

### Parameter Default

```python
# Parameter dengan default value
def sapa(nama, pesan="Selamat datang"):
    print(f"{pesan}, {nama}!")

sapa("Faaza")                    # Selamat datang, Faaza!
sapa("Faaza", "Selamat pagi")    # Selamat pagi, Faaza!

# Multiple parameter dengan default
def hitung_luas(panjang, lebar, unit="meter"):
    luas = panjang * lebar
    return f"{luas} {unit} persegi"

print(hitung_luas(10, 5))              # 50 meter persegi
print(hitung_luas(10, 5, "cm"))        # 50 cm persegi
```

### *args dan **kwargs

Untuk menerima jumlah parameter yang tidak tetap.

```python
# *args - arbitrary positional arguments (tuple)
def print_semua(*args):
    for item in args:
        print(item)

print_semua(1, 2, 3, "hello")  # Print 1, 2, 3, hello

# **kwargs - arbitrary keyword arguments (dictionary)
def print_kwds(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_kwds(nama="Faaza", umur=25, kota="Jakarta")
# Output:
# nama: Faaza
# umur: 25
# kota: Jakarta

# Kombinasi
def fungsi_kompleks(a, b, *args, **kwargs):
    print(f"a={a}, b={b}")
    print(f"args: {args}")
    print(f"kwargs: {kwargs}")

fungsi_kompleks(1, 2, 3, 4, 5, nama="Faaza", umur=25)
# Output:
# a=1, b=2
# args: (3, 4, 5)
# kwargs: {'nama': 'Faaza', 'umur': 25}
```

### Scope (Variable Scope)

Scope menentukan di mana variable bisa diakses.

```python
x = "global"

def fungsi():
    x = "local"  # local variable
    print(x)     # local

fungsi()      # local
print(x)      # global (tidak berubah)

# Global keyword - ubah variable global
counter = 0

def increment():
    global counter
    counter += 1

increment()
print(counter)  # 1
increment()
print(counter)  # 2
```

---

## ⚡ Lambda Functions

Lambda adalah anonymous function untuk operasi sederhana.

```python
# Fungsi biasa
def kuadrat(x):
    return x ** 2

# Lambda equivalent
kuadrat_lambda = lambda x: x ** 2

print(kuadrat(5))          # 25
print(kuadrat_lambda(5))   # 25

# Lambda dengan multiple parameters
tambah = lambda x, y: x + y
print(tambah(5, 3))  # 8

# Lambda dengan kondisi
kategori = lambda umur: "Dewasa" if umur >= 18 else "Remaja"
print(kategori(20))  # Dewasa
print(kategori(15))  # Remaja
```

### Lambda dengan Built-in Functions

Lambda sering digunakan dengan `map()`, `filter()`, `sorted()`.

```python
angka = [1, 2, 3, 4, 5]

# map() - apply function ke setiap elemen
kuadrat = list(map(lambda x: x ** 2, angka))
print(kuadrat)  # [1, 4, 9, 16, 25]

# filter() - filter elements yang memenuhi kondisi
genap = list(filter(lambda x: x % 2 == 0, angka))
print(genap)  # [2, 4]

# sorted() - sort dengan custom key
data = [("Faaza", 25), ("Budi", 20), ("Citra", 23)]
sorted_data = sorted(data, key=lambda x: x[1])  # sort by umur
print(sorted_data)
# [('Budi', 20), ('Citra', 23), ('Faaza', 25)]
```

---

## 📋 List Comprehension

List comprehension adalah cara singkat membuat list baru dari list yang ada.

### Basic List Comprehension

```python
# Cara tradisional dengan loop
angka = [1, 2, 3, 4, 5]
kuadrat = []
for x in angka:
    kuadrat.append(x ** 2)
print(kuadrat)  # [1, 4, 9, 16, 25]

# Cara modern - list comprehension (1 line!)
kuadrat = [x ** 2 for x in angka]
print(kuadrat)  # [1, 4, 9, 16, 25]

# Format: [expression for item in iterable]
```

### List Comprehension dengan Kondisi

```python
angka = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter - hanya angka genap
genap = [x for x in angka if x % 2 == 0]
print(genap)  # [2, 4, 6, 8, 10]

# Transform dan filter
hasil = [x * 2 for x in angka if x > 5]
print(hasil)  # [12, 14, 16, 18, 20]

# Multiple conditions
hasil = [x for x in angka if x > 3 if x < 8]
print(hasil)  # [4, 5, 6, 7]
```

### List Comprehension dengan Expression Kompleks

```python
# If-else dalam expression
angka = [1, 2, 3, 4, 5]
hasil = [x if x % 2 == 0 else x * 2 for x in angka]
print(hasil)  # [2, 2, 6, 4, 10]

# Nested loops dalam comprehension
matriks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [x for baris in matriks for x in baris]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# String manipulation
words = ["hello", "world", "python"]
lengths = [len(w) for w in words]
print(lengths)  # [5, 5, 6]

uppercase = [w.upper() for w in words]
print(uppercase)  # ['HELLO', 'WORLD', 'PYTHON']
```

### Dictionary dan Set Comprehension

```python
# Dictionary comprehension
angka = [1, 2, 3, 4, 5]
kuadrat_dict = {x: x**2 for x in angka}
print(kuadrat_dict)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Dict comprehension dengan kondisi
genap_dict = {x: x**2 for x in angka if x % 2 == 0}
print(genap_dict)  # {2: 4, 4: 16}

# Set comprehension
unique_kuadrat = {x**2 for x in [1, 2, 2, 3, 3, 4]}
print(unique_kuadrat)  # {1, 4, 9, 16}
```

---

## 🎯 Function Best Practices

### Docstring

Dokumentasi untuk function.

```python
def hitung_bmi(berat, tinggi):
    """
    Menghitung BMI (Body Mass Index).
    
    Parameters:
    berat (float): Berat badan dalam kg
    tinggi (float): Tinggi badan dalam meter
    
    Returns:
    float: BMI value
    
    Example:
    >>> hitung_bmi(70, 1.75)
    22.857...
    """
    return berat / (tinggi ** 2)

# Access docstring
print(hitung_bmi.__doc__)
help(hitung_bmi)
```

### Type Hints

Optional - untuk clarity dan IDE support.

```python
def tambah(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def greet(nama: str, umur: int) -> str:
    """Greet someone"""
    return f"Halo {nama}, umur {umur} tahun"

# Type hints tidak enforce (Python is dynamically typed)
print(tambah(5, 3))        # 8
print(tambah("5", "3"))    # 53 (string concatenation - no error!)
```

### Function Composition

Kombinasi functions.

```python
def double(x):
    return x * 2

def add_one(x):
    return x + 1

# Compose manually
x = 5
result = double(add_one(x))  # (5 + 1) * 2 = 12
print(result)

# Or dengan pipeline (functools)
from functools import reduce

def compose(*functions):
    def inner(x):
        for f in reversed(functions):
            x = f(x)
        return x
    return inner

pipeline = compose(add_one, double)
print(pipeline(5))  # double(add_one(5)) = 12
```

---

## 📝 Praktik: Mini Project

### Validator Function

```python
def validate_email(email):
    """Validate email format"""
    if "@" not in email or "." not in email:
        return False
    parts = email.split("@")
    if len(parts) != 2:
        return False
    return True

# Test
print(validate_email("faaza@email.com"))   # True
print(validate_email("faaza@email"))       # False
print(validate_email("faaza.email.com"))   # False
```

### Data Processing Pipeline

```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Step 1: Filter genap
genap = [x for x in data if x % 2 == 0]

# Step 2: Kuadratkan
kuadrat = [x**2 for x in genap]

# Step 3: Sort descending
hasil = sorted(kuadrat, reverse=True)

print(hasil)  # [100, 64, 36, 16, 4]

# Atau dengan comprehension one-liner
hasil = sorted([x**2 for x in data if x % 2 == 0], reverse=True)
print(hasil)
```

---

## 📝 Ringkasan Halaman Ini

### Function Syntax

```python
def nama_fungsi(param1, param2=default):
    """Docstring"""
    # function body
    return result
```

### List Comprehension Syntax

```python
[expression for item in iterable if condition]
```

---

## ✏️ Latihan

### Latihan 1: Function dengan Parameter

```python
def kategori_umur(umur):
    """Kategorize umur"""
    if umur < 13:
        return "Anak-anak"
    elif umur < 20:
        return "Remaja"
    elif umur < 60:
        return "Dewasa"
    else:
        return "Lansia"

# Test
print(kategori_umur(10))   # Anak-anak
print(kategori_umur(15))   # Remaja
print(kategori_umur(30))   # Dewasa
print(kategori_umur(70))   # Lansia
```

### Latihan 2: List Comprehension

```python
# Generate list 1-100, filter odd numbers, multiply by 2
data = [x * 2 for x in range(1, 101) if x % 2 == 1]
print(f"First 5: {data[:5]}")
print(f"Last 5: {data[-5:]}")

# Count
print(f"Total: {len(data)}")
```

### Latihan 3: Kombinasi

```python
# Function yang return multiple values
def statistik_list(data):
    """Return mean, median, min, max"""
    return (
        sum(data) / len(data),
        sorted(data)[len(data) // 2],
        min(data),
        max(data)
    )

# Test
angka = [5, 10, 15, 20, 25]
mean, median, minimum, maximum = statistik_list(angka)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Min: {minimum}")
print(f"Max: {maximum}")
```

---

## ❓ FAQ

### Q: Kapan pakai lambda vs regular function?

**A:**
- **Lambda** - simple one-liner operations
- **Function** - logic lebih kompleks, reusable, dengan documentation

```python
# Lambda: OK
double = lambda x: x * 2

# Function: lebih baik
def calculate_discount(price, discount_percent):
    """Calculate price after discount"""
    return price * (1 - discount_percent / 100)
```

### Q: Apa bedanya list comprehension vs map/filter?

**A:**
- **List comprehension** - lebih Pythonic, readable
- **map/filter** - functional programming style

```python
# List comprehension
result = [x**2 for x in range(10) if x % 2 == 0]

# map/filter
result = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(10))))

# List comprehension lebih jelas!
```

### Q: Bisa nesting list comprehension?

**A:** Ya, tapi lebih sulit dibaca:
```python
# Flattening 2D list
matrix = [[1,2,3], [4,5,6], [7,8,9]]
flattened = [x for row in matrix for x in row]
print(flattened)  # [1,2,3,4,5,6,7,8,9]

# Transpose matrix
transposed = [[row[i] for row in matrix] for i in range(3)]
print(transposed)  # [[1,4,7], [2,5,8], [3,6,9]]
```

---

## 🔗 Referensi

- [Python Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
- [List Comprehensions](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions)
- [Lambda Functions](https://www.w3schools.com/python/python_lambda.asp)
