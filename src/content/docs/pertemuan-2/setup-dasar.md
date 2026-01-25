---
title: Setup & Dasar Python
description: Memulai dengan Python dan environment setup
sidebar:
  order: 2
---

## 🚀 Memulai Python

![Python Programming](https://images.unsplash.com/photo-1526379095098-d400fd0bf935?w=800&h=400&fit=crop)
_Ilustrasi: Python adalah bahasa pemrograman paling populer untuk Data Science_

### Mengapa Python untuk Data Science?

Python adalah bahasa pilihan utama untuk Data Science karena:

1. **Mudah Dipelajari** - Sintaksis yang bersih dan readable
2. **Library Lengkap** - Pandas, NumPy, Scikit-learn, TensorFlow
3. **Komunitas Besar** - Banyak resources dan support
4. **Versatile** - Bisa untuk web, automation, AI, dll

---

## 💻 Environment yang Bisa Digunakan

### Pilihan Environment

| Environment      | Pros             | Cons                 | Untuk Siapa |
| ---------------- | ---------------- | -------------------- | ----------- |
| Google Colab     | Gratis, no setup, GPU gratis | Butuh internet | **Pemula** |
| Jupyter Notebook | Interactive, local | Perlu install | Intermediate |
| VS Code          | Full IDE, professional | Setup kompleks | Developer |
| PyCharm          | Powerful IDE | Heavy, paid version | Professional |

---

## 🎯 Rekomendasi untuk Pemula: Google Colab

### Mengapa Google Colab?

✅ **Tidak perlu install** - Buka di browser, langsung bisa coding  
✅ **GPU/TPU gratis** - Untuk machine learning nanti  
✅ **Terintegrasi Google Drive** - Simpan notebook otomatis  
✅ **Semua library sudah ada** - NumPy, Pandas, Scikit-learn, TensorFlow  
✅ **Share mudah** - Like Google Docs, bisa collaborate  

### Cara Mulai dengan Google Colab

**Step 1: Buka Colab**
- Kunjungi [colab.research.google.com](https://colab.research.google.com)
- Login dengan Google account kamu

**Step 2: Buat Notebook Baru**
- Klik `File` → `New Notebook`
- Atau buka existing notebook dari Drive

**Step 3: Mulai Coding!**
- Ketik code di cell
- Tekan `Shift + Enter` untuk run
- Lihat output di bawah cell

### Contoh Cell Pertama

```python
# Run code pertama kamu!
print("Halo Python!")
print("Saya belajar Data Science!")

# Math operations
hasil = 5 + 3
print(f"5 + 3 = {hasil}")
```

---

## 🛠️ Setup Local (Optional)

Jika ingin setup di laptop/PC:

### Step 1: Install Python

**Windows/Mac/Linux:**
- Download dari [python.org](https://www.python.org/downloads)
- Jangan lupa centang "Add Python to PATH" saat install

**Verify installation:**
```bash
python --version  # atau python3 --version
```

### Step 2: Install Jupyter (Optional)

```bash
pip install jupyter
```

Run Jupyter:
```bash
jupyter notebook
```

Akan membuka di browser secara otomatis.

### Step 3: Install Essential Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🔧 Basic Python Syntax

### Hello World

```python
print("Hello World!")  # Output: Hello World!
```

### Comments

```python
# Ini adalah single-line comment

"""
Ini adalah multi-line comment
atau docstring
bisa multiple line
"""
```

### Variables dan Assignment

```python
# Variable assignment
name = "Faaza"
age = 25
height = 175.5
is_student = True

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0  # semua jadi 0
```

---

## 📊 Basic Input & Output

### Print Output

```python
# Print sederhana
print("Hello")

# Print multiple values
print("Nama:", "Faaza", "Umur:", 25)

# Formatting
print(f"Nama saya {name} dan umur saya {age} tahun")
```

### Input dari User

```python
# Input selalu string
nama = input("Masukkan nama kamu: ")
print(f"Halo {nama}!")

# Convert ke integer
umur = int(input("Masukkan umur kamu: "))
print(f"Tahun depan umur kamu {umur + 1}")
```

---

## 🔍 Mengecek Tipe Data

### Type Function

```python
# type() untuk mengecek tipe data
nama = "Faaza"
umur = 25
tinggi = 175.5
is_student = True

print(type(nama))        # <class 'str'>
print(type(umur))        # <class 'int'>
print(type(tinggi))      # <class 'float'>
print(type(is_student))  # <class 'bool'>

# Type checking dalam conditional
if isinstance(umur, int):
    print(f"{umur} adalah integer")
```

---

## 🔄 Konversi Tipe Data

### Type Casting

```python
# String ke Integer
angka_str = "100"
angka_int = int(angka_str)
print(angka_int + 50)  # 150

# Integer ke String
umur = 25
pesan = "Umur saya " + str(umur) + " tahun"
print(pesan)  # Umur saya 25 tahun

# Float ke Integer (dibulatkan ke bawah)
nilai = 9.7
print(int(nilai))  # 9

# String ke Float
nilai_str = "3.14"
nilai_float = float(nilai_str)
print(nilai_float * 2)  # 6.28

# Ke Boolean
print(bool(1))      # True
print(bool(0))      # False
print(bool(""))     # False (string kosong)
print(bool("hello")) # True (string tidak kosong)
```

---

## 📈 Basic Math Operations

```python
# Aritmatika
x = 10
y = 3

print(x + y)    # 13
print(x - y)    # 7
print(x * y)    # 30
print(x / y)    # 3.333...
print(x // y)   # 3 (integer division)
print(x % y)    # 1 (modulo/remainder)
print(x ** y)   # 1000 (exponential)

# Assignment operators
x += 5   # x = x + 5 (same as x = 15)
x -= 3   # x = x - 3
x *= 2   # x = x * 2
x /= 4   # x = x / 4
```

---

## 💡 Useful Built-in Functions

```python
# Numeric functions
print(abs(-5))          # 5 (absolute value)
print(max(3, 8, 2))     # 8
print(min(3, 8, 2))     # 2
print(round(3.7))       # 4
print(pow(2, 3))        # 8 (2^3)
print(sum([1, 2, 3]))   # 6

# String functions
text = "hello world"
print(len(text))        # 11 (length)
print(text.upper())     # HELLO WORLD
print(text.capitalize()) # Hello world
```

---

## 📝 Ringkasan Halaman Ini

### Key Points

| Konsep | Contoh |
| ------ | ------- |
| Print | `print("Hello")` |
| Variable | `nama = "Faaza"` |
| Type | `type(25)` → `<class 'int'>` |
| Casting | `int("25")` → `25` |
| Math | `5 + 3`, `10 / 2`, `2 ** 3` |

---

## ✏️ Latihan

### Latihan 1: Setup Environment

1. Pilih environment (Google Colab atau local)
2. Jalankan code berikut:

```python
print("✅ Environment ready!")
print(f"Python version check passed")
```

### Latihan 2: Variables & Types

```python
# Buat variabel-variabel berikut
nama = "Nama Kamu"
umur = 20  # ganti dengan umur kamu
tinggi_cm = 170.5

# Print dengan f-string
print(f"Nama saya {nama}")
print(f"Umur saya {umur} tahun")
print(f"Tinggi saya {tinggi_cm} cm")

# Cek tipe data
print(f"Tipe data tinggi: {type(tinggi_cm)}")
```

### Latihan 3: Input & Conversion

```python
# Input dari user
nama_user = input("Siapa nama kamu? ")
umur_user = int(input("Berapa umur kamu? "))

# Compute
tahun_depan = umur_user + 1

# Output
print(f"Halo {nama_user}!")
print(f"Tahun depan umur kamu {tahun_depan}")
```

---

## ❓ FAQ

### Q: Apa bedanya Python 2 dan Python 3?

**A:** Python 2 sudah discontinued (akhir support 2020). Gunakan Python 3 (3.8+). Jangan khawatir, semua course modern menggunakan Python 3.

### Q: Kenapa print function, bukan statement?

**A:** Di Python 3, `print()` adalah function (gunakan parenthesis). Di Python 2, bisa `print` tanpa parenthesis. Python 3 lebih consistent.

### Q: Google Colab vs Jupyter, mana lebih baik?

**A:** Untuk pemula: **Google Colab**. Tidak perlu setup, gratis GPU. Begitu lebih expert, pindah ke Jupyter atau VS Code untuk kontrol lebih.

### Q: Apa itu f-string?

**A:** Format string dengan `f"text {variable}"`. Lebih modern dan readable dibanding `.format()` atau `%` operator.

---

## 🔗 Referensi

- [Python Official Documentation](https://docs.python.org/)
- [Google Colab Tutorial](https://colab.research.google.com/notebooks/intro.ipynb)
- [Python w3schools - Beginner](https://www.w3schools.com/python/)
