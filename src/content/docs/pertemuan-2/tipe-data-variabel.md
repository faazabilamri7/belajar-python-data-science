---
title: Tipe Data & Variabel
description: Memahami tipe data dasar dan manipulasi string
sidebar:
  order: 3
---

## 📦 Variabel dan Tipe Data

### Apa itu Variabel?

Variabel adalah "wadah" atau "tempat" untuk menyimpan data. Bayangkan seperti kotak berlabel dimana kamu bisa menyimpan berbagai jenis barang.

```python
# Membuat variabel - gunakan nama yang meaningful
nama = "Faaza"           # menyimpan text
umur = 25                # menyimpan angka bulat
tinggi = 175.5           # menyimpan angka desimal
is_student = True        # menyimpan boolean (true/false)
```

### Aturan Naming Variable

```python
# ✅ Valid names
my_name = "Faaza"
myName = "Faaza"          # camelCase (less Pythonic)
my_name_is = "Faaza"
_private_var = 100
__magic__ = "special"

# ❌ Invalid names
2my_name = "Faaza"        # tidak boleh mulai dengan angka
my-name = "Faaza"         # tidak boleh dash
my name = "Faaza"         # tidak boleh spasi
class = "Data Science"    # class adalah keyword
```

**Rekomendasi:** Gunakan `snake_case` (pisah dengan underscore) - ini adalah Python convention.

---

## 🔢 Tipe Data Dasar

### Integer (int)

Bilangan bulat tanpa desimal.

```python
umur = 25
tahun = 2024
jumlah_peserta = 100
negatif = -50
nol = 0

print(type(25))  # <class 'int'>

# Operasi integer
print(10 + 5)    # 15
print(10 - 5)    # 5
print(10 * 5)    # 50
print(10 / 5)    # 2.0 (hasil float!)
print(10 // 5)   # 2 (integer division)
print(10 % 3)    # 1 (modulo)
print(2 ** 3)    # 8 (exponential)
```

### Float (float)

Bilangan dengan desimal (floating point).

```python
tinggi = 175.5
berat = 70.2
pi = 3.14159
suhu = -5.5
scientific = 1.5e-3  # 0.0015 (scientific notation)

print(type(3.14))  # <class 'float'>

# Operasi float
print(10.5 + 5.2)  # 15.7
print(10.5 * 2)    # 21.0
```

### String (str)

Text atau teks - urutan characters.

```python
nama = "Faaza"
judul = 'Data Science'
alamat = """Jl. Merdeka No. 123
Jakarta Pusat"""  # multi-line string

print(type("hello"))  # <class 'str'>

# Empty string
text_kosong = ""
```

### Boolean (bool)

Hanya dua nilai: `True` atau `False`.

```python
is_active = True
is_completed = False
has_permission = True

print(type(True))  # <class 'bool'>

# Boolean dari comparison
print(5 > 3)       # True
print(5 < 3)       # False
print(5 == 5)      # True
```

---

## 🔄 Mengecek dan Konversi Tipe Data

### Type Checking

```python
nama = "Faaza"
umur = 25
tinggi = 175.5
is_student = True

# Method 1: type()
print(type(nama))        # <class 'str'>
print(type(umur))        # <class 'int'>
print(type(tinggi))      # <class 'float'>
print(type(is_student))  # <class 'bool'>

# Method 2: isinstance()
if isinstance(umur, int):
    print(f"{umur} adalah integer")

if isinstance(nama, str):
    print(f"{nama} adalah string")
```

### Type Conversion (Casting)

```python
# String ke Integer
angka_str = "100"
angka_int = int(angka_str)  # 100
print(angka_int + 50)       # 150

# Integer ke String
umur = 25
pesan = "Umur saya " + str(umur) + " tahun"
print(pesan)  # Umur saya 25 tahun

# Float ke Integer (dipotong/truncated, bukan rounded)
nilai = 9.7
print(int(nilai))   # 9 (bukan 10!)

# Integer ke Float
angka = 10
print(float(angka))  # 10.0

# String ke Float
harga_str = "15.50"
harga_float = float(harga_str)  # 15.5
print(harga_float * 2)          # 31.0

# Ke Boolean
print(bool(1))          # True
print(bool(0))          # False
print(bool(-5))         # True (semua non-zero = True)
print(bool(""))         # False (string kosong)
print(bool("hello"))    # True (string tidak kosong)
print(bool([]))         # False (list kosong)
```

---

## 📝 String Operations

### String Basics

```python
teks = "Hello World"

# Length
print(len(teks))  # 11

# Access individual character (indexing)
print(teks[0])    # H (karakter pertama)
print(teks[6])    # W (karakter ke-7)
print(teks[-1])   # d (karakter terakhir)
print(teks[-2])   # l (karakter kedua dari belakang)
```

### String Concatenation dan Repetition

```python
nama_depan = "Faaza"
nama_belakang = "Bil Amri"

# Concatenation (penggabungan)
nama_lengkap = nama_depan + " " + nama_belakang
print(nama_lengkap)  # Faaza Bil Amri

# Repetition (pengulangan)
garis = "-" * 20
print(garis)  # -------------------- (20 dash)

pemisah = "=" * 5
print(pemisah)  # =====

# Combine
bingkai = "-" * 10 + " Title " + "-" * 10
print(bingkai)  # ---------- Title ----------
```

### String Indexing & Slicing

```python
teks = "Python"

# Indexing - ambil character tertentu
print(teks[0])    # P
print(teks[1])    # y
print(teks[-1])   # n (last)

# Slicing - ambil range of characters [start:end]
print(teks[0:3])  # Pyt (index 0, 1, 2 - bukan 3!)
print(teks[1:4])  # yth
print(teks[2:])   # thon (dari index 2 sampai akhir)
print(teks[:4])   # Pyth (dari awal sampai index 3)
print(teks[:])    # Python (semua)
print(teks[::2])  # Pto (setiap 2 character)
print(teks[::-1]) # nohtyP (reverse!)
```

### String Methods

```python
teks = "  Hello World  "

# Case
print(teks.lower())      # "  hello world  "
print(teks.upper())      # "  HELLO WORLD  "
print(teks.capitalize()) # "  hello world  " (capitalize first char)
print(teks.title())      # "  Hello World  " (capitalize each word)

# Whitespace
print(teks.strip())      # "Hello World" (remove leading/trailing spaces)
print(teks.lstrip())     # "Hello World  " (remove left spaces only)
print(teks.rstrip())     # "  Hello World" (remove right spaces only)

# Search & Replace
print("World" in teks)   # True (check if substring exists)
print(teks.find("World"))  # 8 (index of substring, -1 if not found)
print(teks.replace("World", "Python"))  # "  Hello Python  "
print(teks.count("l"))   # 3 (count occurrences)

# Split & Join
print(teks.split())      # ['Hello', 'World'] (split by whitespace)
print("a,b,c".split(","))  # ['a', 'b', 'c']
print("-".join(['a', 'b', 'c']))  # "a-b-c"

# Startswith & Endswith
print("Hello".startswith("He"))  # True
print("Hello".endswith("lo"))    # True
```

### String Formatting

Ada beberapa cara untuk format string:

```python
nama = "Faaza"
umur = 25
tinggi = 175.5

# Method 1: f-string (Python 3.6+) - RECOMMENDED
print(f"Nama: {nama}, Umur: {umur}, Tinggi: {tinggi}")
# Output: Nama: Faaza, Umur: 25, Tinggi: 175.5

# Dengan formatting
print(f"Tinggi: {tinggi:.2f} cm")  # Tinggi: 175.50 cm
print(f"Umur: {umur:03d}")  # Umur: 025 (3 digit, pad with 0)

# Method 2: .format() method
print("Nama: {}, Umur: {}".format(nama, umur))
print("Nama: {0}, Umur: {1}, Nama lagi: {0}".format(nama, umur))

# Method 3: % operator (older style, tidak recommended)
print("Nama: %s, Umur: %d" % (nama, umur))
```

---

## 🎯 Useful String Functions

```python
teks = "Hello World 123"

# Type checking
print(teks.isalpha())      # False (ada angka)
print(teks.isdigit())      # False (ada text)
print(teks.isalnum())      # False (ada spasi)
print("123".isdigit())     # True (semua digit)
print("hello".islower())   # True (semua lowercase)
print("HELLO".isupper())   # True (semua uppercase)

# Remove characters
print(teks.strip())        # "Hello World 123"
```

---

## 📝 Ringkasan Halaman Ini

### Tipe Data Primitif

| Tipe | Contoh | Deskripsi |
| ---- | ------ | --------- |
| int | `25`, `-10`, `0` | Bilangan bulat |
| float | `3.14`, `-0.5` | Bilangan desimal |
| str | `"Hello"`, `'World'` | Text/string |
| bool | `True`, `False` | Boolean |

### String Methods Penting

| Method | Hasil |
| ------ | ----- |
| `.lower()`, `.upper()` | Konversi case |
| `.strip()` | Hapus whitespace |
| `.split()`, `.join()` | Split/join text |
| `.replace()` | Replace substring |
| `in` operator | Check substring |

---

## ✏️ Latihan

### Latihan 1: Variabel & Tipe Data

```python
# Buat variabel untuk data pribadi kamu
nama = "Nama Kamu"
umur = 20
tinggi = 170.5
is_student = True

# Check type
print(f"Type nama: {type(nama)}")
print(f"Type umur: {type(umur)}")

# Print dengan f-string
print(f"\nProfile:")
print(f"Nama: {nama}")
print(f"Umur: {umur}")
print(f"Tinggi: {tinggi:.1f} cm")
print(f"Status: {'Mahasiswa' if is_student else 'Bukan Mahasiswa'}")
```

### Latihan 2: String Manipulation

```python
# Input
nama = "faaza bil amri"

# Manipulate
nama_upper = nama.upper()
nama_title = nama.title()
nama_len = len(nama)

# Print
print(f"Original: {nama}")
print(f"Uppercase: {nama_upper}")
print(f"Title case: {nama_title}")
print(f"Length: {nama_len}")

# Slicing
print(f"First 5 chars: {nama[:5]}")
print(f"Last 3 chars: {nama[-3:]}")
print(f"Every 2 chars: {nama[::2]}")
```

### Latihan 3: Type Conversion

```python
# Input dari user
nama = input("Siapa nama kamu? ")
umur_str = input("Berapa umur kamu? ")
tinggi_str = input("Berapa tinggi kamu (cm)? ")

# Convert
umur = int(umur_str)
tinggi = float(tinggi_str)

# Process
tahun_depan = umur + 1

# Output
print(f"\nProfile {nama}:")
print(f"Umur saat ini: {umur}")
print(f"Umur tahun depan: {tahun_depan}")
print(f"Tinggi: {tinggi} cm")
```

---

## ❓ FAQ

### Q: Apa bedanya single quote vs double quote untuk string?

**A:** Tidak ada bedanya secara fungsional! Gunakan yang konsisten. Convention Python lebih prefer double quote, tapi flexible. Pilih satu dan gunakan consistently.

```python
nama1 = "Faaza"   # double quote
nama2 = 'Faaza'   # single quote
# Sama saja!
```

### Q: Kenapa `int("10.5")` error?

**A:** Karena `"10.5"` adalah string yang berisi desimal. Harus convert ke float dulu:
```python
float("10.5")  # 10.5 - berhasil
int(float("10.5"))  # 10 - baru bisa convert ke int
```

### Q: Bagaimana reverse string?

**A:** Gunakan slicing dengan step negative:
```python
teks = "Hello"
print(teks[::-1])  # olleH
```

### Q: Perbedaan `len()` vs `count()`?

**A:** 
- `len(teks)` - total character (termasuk spasi)
- `teks.count(char)` - berapa kali character tertentu muncul

```python
teks = "Hello"
print(len(teks))        # 5 (5 character total)
print(teks.count("l"))  # 2 (2 huruf 'l')
```

---

## 🔗 Referensi

- [Python String Methods - W3Schools](https://www.w3schools.com/python/python_string_methods.asp)
- [Python Built-in Functions](https://docs.python.org/3/library/functions.html)
