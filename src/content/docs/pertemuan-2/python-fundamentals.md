---
title: Dasar Pemrograman Python
description: Belajar sintaksis Python dasar untuk kebutuhan Data Science
sidebar:
  order: 1
---

## 🎯 Tujuan Pembelajaran

Setelah mempelajari materi ini, kamu akan mampu:

- ✅ Menjalankan kode Python di berbagai environment
- ✅ Memahami tipe data dan variabel
- ✅ Menggunakan operator dan expressions
- ✅ Membuat control flow (if, for, while)
- ✅ Menulis dan menggunakan functions

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

### Environment yang Bisa Digunakan

| Environment      | Pros             | Cons                 |
| ---------------- | ---------------- | -------------------- |
| Google Colab     | Gratis, no setup | Butuh internet       |
| Jupyter Notebook | Interactive      | Perlu install        |
| VS Code          | Full IDE         | Setup lebih kompleks |

:::tip[Rekomendasi untuk Pemula]
Gunakan **Google Colab** untuk memulai. Buka [colab.google](https://colab.research.google.com/), login dengan akun Google, dan kamu siap coding!
:::

---

## 📦 Variabel dan Tipe Data

### Apa itu Variabel?

Variabel adalah "wadah" untuk menyimpan data. Bayangkan seperti kotak berlabel.

```python
# Membuat variabel
nama = "Faaza"
umur = 25
tinggi = 175.5
is_student = True
```

### Tipe Data Dasar

| Tipe    | Contoh               | Deskripsi             |
| ------- | -------------------- | --------------------- |
| `int`   | `25`, `-10`, `0`     | Bilangan bulat        |
| `float` | `3.14`, `-0.5`       | Bilangan desimal      |
| `str`   | `"Hello"`, `'World'` | Teks/string           |
| `bool`  | `True`, `False`      | Boolean (benar/salah) |

### Mengecek Tipe Data

Fungsi `type()` digunakan untuk mengecek jenis data dari variabel. Ini sangat berguna saat kamu melakukan operasi yang membutuhkan tipe data tertentu.

```python
nama = "Faaza"              # String
umur = 25                   # Integer
tinggi = 175.5              # Float
is_student = True           # Boolean

# Menggunakan function type() untuk cek tipe data
print(type(nama))           # <class 'str'> - teks
print(type(umur))           # <class 'int'> - bilangan bulat
print(type(tinggi))         # <class 'float'> - desimal
print(type(is_student))     # <class 'bool'> - true/false

# Penjelasan:
# type() adalah built-in function yang mengembalikan class/tipe dari sebuah value
# Tipe data ini penting karena berbagai operasi hanya bisa dilakukan pada tipe tertentu
# Contoh: kamu tidak bisa menjumlahkan string dengan angka tanpa konversi dulu
```

### Konversi Tipe Data

Kadang kita perlu mengubah tipe data dari satu jenis ke jenis lain. Misalnya input dari user selalu string, tapi kita butuh angka. Berikut cara konversi:

```python
# String ke Integer - mengubah teks berisi angka menjadi angka bulat
angka_str = "100"           # ini string (tipe text)
angka_int = int(angka_str)  # int() convert string menjadi integer
print(angka_int + 50)       # 150 - sekarang bisa di-operasi dengan angka

# Integer ke String - mengubah angka menjadi teks
umur = 25                   # ini integer
pesan = "Umur saya " + str(umur) + " tahun"  # str() convert int ke string
print(pesan)                # Umur saya 25 tahun - bisa digabung dengan string

# Float ke Integer - mengubah desimal ke bilangan bulat (DIBULATKAN KE BAWAH)
nilai = 9.7                 # ini float
print(int(nilai))           # 9 - desimalnya dihapus (tidak dibulatkan, tapi dipotong)

# Penjelasan:
# int() - convert ke integer
# str() - convert ke string
# float() - convert ke float
# bool() - convert ke boolean (0=False, sisanya=True)
```

---

## 📝 String (Teks)

### Membuat String

```python
# Dengan kutip tunggal atau ganda
nama1 = 'Faaza'
nama2 = "Faaza"

# String panjang (multi-line)
paragraf = """
Ini adalah paragraf
yang sangat panjang
dan multi-baris
"""
```

### Operasi String

String bisa dilakukan berbagai operasi seperti penggabungan, pengulangan, dan measuring.

```python
nama_depan = "Faaza"
nama_belakang = "Bil Amri"

# Concatenation (penggabungan) - menggabung 2 atau lebih string dengan +
nama_lengkap = nama_depan + " " + nama_belakang
print(nama_lengkap)  # Faaza Bil Amri - hasil penggabungan 3 string

# Repetition (pengulangan) - mengulangi string dengan *
garis = "-" * 20  # karakter "-" diulang 20 kali
print(garis)  # -------------------- - berguna untuk membuat separator

# Length (panjang) - menghitung berapa banyak karakter dalam string
print(len(nama_lengkap))  # 14 - termasuk spasi

# Penjelasan:
# + untuk concat (hanya untuk string ke string, tidak bisa string + angka)
# * untuk repeat (angka * string atau string * angka)
# len() menghitung jumlah karakter (termasuk spasi dan karakter khusus)
```

### String Formatting

```python
nama = "Faaza"
umur = 25

# f-string (recommended - Python 3.6+)
pesan = f"Halo, nama saya {nama} dan umur saya {umur} tahun"
print(pesan)

# format method
pesan2 = "Halo, nama saya {} dan umur saya {} tahun".format(nama, umur)
print(pesan2)
```

### String Methods

```python
teks = "  Hello World  "

print(teks.lower())      # "  hello world  "
print(teks.upper())      # "  HELLO WORLD  "
print(teks.strip())      # "Hello World" (hapus spasi)
print(teks.replace("World", "Python"))  # "  Hello Python  "
print(teks.split())      # ['Hello', 'World']
```

---

## 📊 Koleksi Data

### Membuat List

List adalah kumpulan data yang urutan-nya penting dan bisa diubah (mutable).

```python
# Membuat list - data disimpan dalam [] dipisah koma
buah = ["apel", "jeruk", "mangga", "pisang"]
angka = [1, 2, 3, 4, 5]
campuran = ["teks", 10, 3.14, True]  # list bisa campur berbagai tipe data

# Mengakses elemen - gunakan index (mulai dari 0, bukan 1!)
print(buah[0])    # apel - elemen pertama
print(buah[1])    # jeruk - elemen kedua
print(buah[-1])   # pisang - index negatif mengakses dari belakang (-1=terakhir)

# Slicing - mengambil range elemen dengan [start:end]
print(buah[1:3])  # ['jeruk', 'mangga'] - dari index 1 sampai 2 (3 tidak termasuk)
print(buah[:2])   # ['apel', 'jeruk'] - dari awal sampai index 1
print(buah[2:])   # ['mangga', 'pisang'] - dari index 2 sampai akhir

# Penjelasan:
# Indexing dimulai dari 0 (tidak 1!)
# Range/slice [start:end] include start tapi exclude end
# Negative index: -1=last, -2=second last, dst
```

### Operasi List

List punya banyak method (function) bawaan untuk memanipulasi data.

```python
buah = ["apel", "jeruk", "mangga"]

# Menambah elemen
buah.append("anggur")           # append() tambah 1 elemen di akhir
buah.insert(1, "semangka")      # insert() tambah di posisi tertentu

# Menghapus elemen
buah.remove("jeruk")            # remove() hapus berdasarkan nilai
buah.pop()                      # pop() hapus elemen terakhir
del buah[0]                     # del statement hapus berdasarkan index

# Operasi lainnya
print(len(buah))                # len() menghitung jumlah elemen
print("apel" in buah)           # in operator cek apakah ada elemen
buah.sort()                     # sort() urutkan elemen (ascending)

# Penjelasan:
# append() - menambah satu elemen ke akhir list
# insert(index, value) - menambah elemen pada posisi tertentu
# remove(value) - menghapus elemen berdasarkan nilai (hapus yang pertama ketemu)
# pop() - menghapus elemen terakhir (bisa juga pop(index))
# del list[index] - statement khusus untuk menghapus berdasarkan index
# in - operator untuk cek keberadaan elemen dalam list
```

### 2. Dictionary (Kamus)

Dictionary menyimpan data dalam format key-value pairs - seperti kamus fisik yang menghubungkan kata (key) dengan definisinya (value).

```python
# Membuat dictionary dengan {} dan key:value pairs
mahasiswa = {
    "nama": "Faaza",           # key "nama" dengan value "Faaza"
    "umur": 25,                # key "umur" dengan value 25
    "jurusan": "Sistem Informasi",
    "ipk": 3.75
}

# Mengakses value dengan key (bukan index!)
print(mahasiswa["nama"])        # Faaza - akses dengan key dalam []
print(mahasiswa.get("umur"))    # 25 - alternatif pakai .get() method

# Menambah/mengubah value
mahasiswa["alamat"] = "Jakarta"  # tambah key baru
mahasiswa["ipk"] = 3.80          # update value dari key yang sudah ada

# Menghapus
del mahasiswa["alamat"]          # hapus key-value pair

# Iterasi - looping untuk access semua key-value
for key, value in mahasiswa.items():
    print(f"{key}: {value}")

# Penjelasan:
# Dictionary beda dengan list: list akses via index, dict akses via key
# Key harus unique (tidak boleh sama), value bisa duplikat
# .get() method lebih aman daripada [] karena tidak error jika key tidak ada
# .items() return semua key-value pairs untuk di-loop
```

### 3. Tuple

Tuple mirip list tapi **immutable** (tidak bisa diubah setelah dibuat).

```python
koordinat = (10.5, 20.3)
rgb = (255, 128, 0)

# Mengakses
print(koordinat[0])  # 10.5

# Unpacking
x, y = koordinat
print(f"x = {x}, y = {y}")
```

---

## 🔢 Operator

### Operator Aritmatika

```python
a = 10
b = 3

print(a + b)   # 13 (penjumlahan)
print(a - b)   # 7  (pengurangan)
print(a * b)   # 30 (perkalian)
print(a / b)   # 3.333... (pembagian)
print(a // b)  # 3  (pembagian bulat)
print(a % b)   # 1  (modulo/sisa bagi)
print(a ** b)  # 1000 (pangkat)
```

### Operator Perbandingan

```python
x = 5
y = 10

print(x == y)  # False (sama dengan)
print(x != y)  # True  (tidak sama dengan)
print(x < y)   # True  (kurang dari)
print(x > y)   # False (lebih dari)
print(x <= y)  # True  (kurang dari atau sama dengan)
print(x >= y)  # False (lebih dari atau sama dengan)
```

### Operator Logika

```python
a = True
b = False

print(a and b)  # False
print(a or b)   # True
print(not a)    # False
```

---

## 🔀 Control Flow

### If-Elif-Else

Conditional statement digunakan untuk membuat keputusan dalam program berdasarkan kondisi tertentu.

```python
nilai = 85  # input nilai ujian

# if-elif-else structure untuk menentukan grade
if nilai >= 90:
    grade = "A"  # jika nilai >= 90, grade A
elif nilai >= 80:
    grade = "B"  # jika tidak, cek apakah >= 80, maka B
elif nilai >= 70:
    grade = "C"
elif nilai >= 60:
    grade = "D"
else:
    grade = "E"  # jika semua kondisi gagal, maka E

print(f"Grade kamu: {grade}")  # Grade kamu: B

# Penjelasan:
# if - condition pertama
# elif (else if) - condition alternatif (bisa multiple)
# else - condition default jika semua if/elif gagal
# elif membantu menghindari nested if yang kompleks
```

### Contoh Praktis

```python
umur = 20
punya_sim = True

if umur >= 17 and punya_sim:
    print("Boleh mengendarai kendaraan")
elif umur >= 17 and not punya_sim:
    print("Silakan buat SIM terlebih dahulu")
else:
    print("Belum cukup umur")
```

---

## 🔁 Loops (Perulangan)

### For Loop

For loop digunakan untuk iterasi (mengulangi) sesuatu sebanyak jumlah tertentu atau untuk setiap item dalam list.

```python
# Iterasi setiap item dalam list
buah = ["apel", "jeruk", "mangga"]
for item in buah:              # untuk setiap item dalam list buah
    print(item)                # print item tersebut

# Menggunakan range() untuk iterasi angka
for i in range(5):             # 0, 1, 2, 3, 4 (total 5 kali)
    print(i)

# range(start, end, step)
for i in range(1, 6):          # 1, 2, 3, 4, 5 (dari 1 sampai 5)
    print(i)

for i in range(0, 10, 2):      # 0, 2, 4, 6, 8 (step/lompat 2)
    print(i)

# Penjelasan:
# range(5) = 0 sampai 4 (5 angka total)
# range(1, 6) = 1 sampai 5 (end tidak termasuk!)
# range(0, 10, 2) = step 2, jadi 0, 2, 4, 6, 8
```

### While Loop

While loop mengulangi block of code selama condition masih True. Beda dengan for loop, while tidak tahu berapa kali akan loop.

```python
counter = 0
while counter < 5:  # selama counter masih kurang dari 5
    print(f"Counter: {counter}")
    counter += 1  # PENTING: counter harus terus dinaikkan, kalau tidak infinite loop!

# Output:
# Counter: 0
# Counter: 1
# Counter: 2
# Counter: 3
# Counter: 4

# Penjelasan:
# while condition:
#   loop body
# Loop akan terus jalan selama condition True
# HATI-HATI: jika tidak ada yang mengubah variable, bisa infinite loop!
```

### Break dan Continue

Break dan continue digunakan untuk control flow dalam loop.

```python
# Break - menghentikan loop seketika
for i in range(10):
    if i == 5:
        break           # keluar dari loop, tidak lanjut ke 6, 7, 8, 9
    print(i)  # output: 0, 1, 2, 3, 4

# Continue - skip iterasi saat ini, lanjut ke iterasi berikutnya
for i in range(5):
    if i == 2:
        continue        # skip angka 2, langsung ke 3
    print(i)  # output: 0, 1, 3, 4

# Penjelasan:
# break - keluar dari loop (tidak lanjut lagi)
# continue - skip iterasi ini, lanjut ke iterasi berikutnya
# Useful untuk skip item tertentu atau stop jika kondisi terpenuhi
```

### List Comprehension

List comprehension adalah cara singkat dan elegant untuk membuat list baru dari list yang ada. Jauh lebih readable daripada loop biasa.

```python
# Cara tradisional dengan loop
angka = [1, 2, 3, 4, 5]
kuadrat = []
for x in angka:
    kuadrat.append(x ** 2)  # hitung kuadrat dan append

# Cara modern dengan list comprehension (1 line!)
kuadrat = [x ** 2 for x in angka]
print(kuadrat)  # [1, 4, 9, 16, 25]

# Dengan kondisi/filter
genap = [x for x in range(10) if x % 2 == 0]
print(genap)  # [0, 2, 4, 6, 8]

# Format: [expression for item in list if condition]
# Contoh kompleks:
hasil = [x * 2 for x in range(10) if x % 2 == 1]  # kalikan 2 angka ganjil 0-9
print(hasil)  # [2, 6, 10, 14, 18]

# Penjelasan:
# [expression for item in iterable] - basic syntax
# expression adalah apa yang mau ditampilkan (bisa x, x**2, f(x), dll)
# for item in iterable - iterasi setiap item
# if condition (optional) - filter hanya item yang match condition
```

---

## 🔧 Functions

### Membuat Function

```python
# Function tanpa parameter
def sapa():
    print("Halo, selamat datang!")

sapa()  # Halo, selamat datang!

# Function dengan parameter
def sapa_nama(nama):
    print(f"Halo, {nama}!")

sapa_nama("Faaza")  # Halo, Faaza!

# Function dengan return value
def tambah(a, b):
    return a + b

hasil = tambah(5, 3)
print(hasil)  # 8
```

### Parameter Default

```python
def sapa(nama, pesan="Selamat datang"):
    print(f"{pesan}, {nama}!")

sapa("Faaza")                    # Selamat datang, Faaza!
sapa("Faaza", "Selamat pagi")    # Selamat pagi, Faaza!
```

### Multiple Return Values

```python
def hitung_statistik(angka):
    total = sum(angka)
    rata_rata = total / len(angka)
    minimum = min(angka)
    maksimum = max(angka)
    return total, rata_rata, minimum, maksimum

data = [10, 20, 30, 40, 50]
total, avg, min_val, max_val = hitung_statistik(data)
print(f"Total: {total}, Rata-rata: {avg}")
```

### Lambda Function

Function anonymous untuk operasi sederhana.

```python
# Function biasa
def kuadrat(x):
    return x ** 2

# Lambda equivalent
kuadrat = lambda x: x ** 2

print(kuadrat(5))  # 25

# Sering digunakan dengan map, filter
angka = [1, 2, 3, 4, 5]
hasil = list(map(lambda x: x ** 2, angka))
print(hasil)  # [1, 4, 9, 16, 25]
```

---

## 📝 Praktik: Mini Project

### Kalkulator Sederhana

```python
def kalkulator(a, b, operasi):
    """
    Kalkulator sederhana

    Parameters:
    a (float): angka pertama
    b (float): angka kedua
    operasi (str): jenis operasi (+, -, *, /)

    Returns:
    float: hasil perhitungan
    """
    if operasi == "+":
        return a + b
    elif operasi == "-":
        return a - b
    elif operasi == "*":
        return a * b
    elif operasi == "/":
        if b != 0:
            return a / b
        else:
            return "Error: Tidak bisa dibagi dengan 0"
    else:
        return "Operasi tidak valid"

# Test
print(kalkulator(10, 5, "+"))  # 15
print(kalkulator(10, 5, "-"))  # 5
print(kalkulator(10, 5, "*"))  # 50
print(kalkulator(10, 5, "/"))  # 2.0
```

---

## 📝 Ringkasan

| Konsep     | Contoh                        |
| ---------- | ----------------------------- |
| Variabel   | `nama = "Faaza"`              |
| Tipe Data  | `int`, `float`, `str`, `bool` |
| List       | `[1, 2, 3]`                   |
| Dictionary | `{"key": "value"}`            |
| If-Else    | `if kondisi: ... else: ...`   |
| For Loop   | `for i in range(5): ...`      |
| Function   | `def nama_fungsi(): ...`      |

---

## ✏️ Latihan

### Latihan 1: Variabel dan Tipe Data

Buat variabel untuk menyimpan data pribadimu (nama, umur, jurusan, IPK) dan tampilkan dalam format yang rapi.

### Latihan 2: List Operations

Buat list nilai ujian 5 mata kuliah, lalu hitung rata-ratanya menggunakan loop.

### Latihan 3: Function

Buat function `hitung_bmi(berat, tinggi)` yang menghitung BMI dan memberikan kategori (underweight, normal, overweight).

---

## ❓ FAQ (Pertanyaan yang Sering Diajukan)

### Q: Kenapa Python menggunakan index mulai dari 0, bukan 1?

**A:** Ini adalah convention dari hampir semua bahasa pemrograman (C, Java, JavaScript, dll). Dimulai dari 0 karena secara teknis, index adalah "offset" dari memory address. Pertama elemen ada di offset 0, yang kedua di offset 1, dst. Kamu akan terbiasa dalam seminggu!

### Q: Apa bedanya append() vs insert()?

**A:**

- `append()` - menambah elemen di akhir list saja
- `insert(index, value)` - menambah di posisi tertentu

Jadi `append()` lebih cepat tapi kurang fleksibel.

### Q: Saya bingung dengan slicing [start:end], mulai dari mana?

**A:** Ingat ini:

- `[1:3]` = dari index 1 sampai index 2 (index 3 TIDAK termasuk)
- `[:3]` = dari awal sampai index 2
- `[2:]` = dari index 2 sampai akhir
- `[:]` = semua elemen

Jadi end-nya selalu excluded!

### Q: Apa itu immutable vs mutable?

**A:**

- **Mutable** (bisa diubah): List, Dictionary - kamu bisa tambah/hapus/ubah elemen
- **Immutable** (tidak bisa diubah): Tuple, String - setelah dibuat tidak bisa berubah

Contoh: `s = "hello"` kamu tidak bisa ubah `s[0]` menjadi 'x'. Tapi list bisa.

### Q: For loop vs While loop, kapan pakai yang mana?

**A:**

- **For loop** - ketika kamu tahu berapa kali harus loop (iterate list, range tertentu)
- **While loop** - ketika kondisi tidak tahu berapa kali (loop sampai kondisi tertentu terpenuhi)

Contoh: `for i in range(5)` kamu tahu 5 kali. Tapi `while user != "quit"` tidak tahu berapa kali user akan input.

### Q: Apa bedanya list comprehension vs regular loop?

**A:** Sama saja hasilnya, tapi list comprehension:

- Lebih singkat dan readable
- Lebih cepat (slightly)
- Lebih Pythonic (idiomatic)

Gunakan list comprehension jika sederhana, kalau logic kompleks gunakan regular loop agar lebih mudah dibaca.

### Q: Function saya error "NameError: name is not defined", apa masalahnya?

**A:** Biasanya tiga kemungkinan:

1. Typo nama variable
2. Variable belum dibuat saat dipakai
3. Variable adalah local variable yang hanya ada dalam function tertentu

Pastikan variable sudah didefinisikan sebelum dipakai!

### Q: Berapa banyak parameter yang bisa diterima function?

**A:** Tidak ada batasan! Kamu bisa bikin function dengan 1, 5, 100 parameter. Tapi lebih dari 5-6 parameter biasanya tanda bahwa function tersebut terlalu kompleks dan perlu di-refactor.

---

:::note[Catatan]
Pertemuan ini adalah fondasi Python. Pastikan kamu memahami semua konsep ini dengan baik sebelum lanjut ke pertemuan berikutnya, karena setiap topik berikutnya bergantung pada materi ini.
