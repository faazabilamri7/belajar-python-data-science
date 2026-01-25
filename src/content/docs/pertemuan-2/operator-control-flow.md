---
title: Operator & Control Flow
description: Operator dan conditional/loop statements
sidebar:
  order: 5
---

## 🔢 Operator

### Operator Aritmatika

```python
a = 10
b = 3

# Basic arithmetic
print(a + b)    # 13 (addition)
print(a - b)    # 7 (subtraction)
print(a * b)    # 30 (multiplication)
print(a / b)    # 3.333... (division - result is float)
print(a // b)   # 3 (floor division - result is int)
print(a % b)    # 1 (modulo - remainder)
print(a ** b)   # 1000 (exponentiation)

# Assignment operators
x = 10
x += 5    # x = x + 5 → x = 15
x -= 3    # x = x - 3 → x = 12
x *= 2    # x = x * 2 → x = 24
x /= 4    # x = x / 4 → x = 6.0
x //= 2   # x = x // 2 → x = 3.0
x %= 2    # x = x % 2 → x = 1.0
x **= 2   # x = x ** 2 → x = 1.0
```

### Operator Perbandingan

Operator perbandingan mengembalikan Boolean (`True` atau `False`).

```python
x = 5
y = 10

# Comparison operators
print(x == y)   # False (sama dengan)
print(x != y)   # True (tidak sama dengan)
print(x < y)    # True (kurang dari)
print(x > y)    # False (lebih dari)
print(x <= y)   # True (kurang dari atau sama dengan)
print(x >= y)   # False (lebih dari atau sama dengan)

# Chaining comparisons
print(5 < 10 < 15)      # True (both conditions true)
print(5 < 10 > 3)       # True (5 < 10 AND 10 > 3)
print(5 == 5.0)         # True (int dan float sama value-nya)
print(5 is 5)           # True (sama object - untuk small integers)
print([1,2] is [1,2])   # False (different objects)
print([1,2] == [1,2])   # True (same content)
```

### Operator Logika

Operator logika menggabungkan kondisi Boolean.

```python
# and - kedua kondisi harus True
print(True and True)    # True
print(True and False)   # False
print(False and False)  # False

# or - minimal salah satu True
print(True or False)    # True
print(False or False)   # False

# not - negasi
print(not True)         # False
print(not False)        # True

# Kombinasi
x = 5
print(x > 0 and x < 10)    # True (5 > 0 AND 5 < 10)
print(x < 0 or x > 10)     # False (5 < 0 OR 5 > 10)
print(not (x > 10))        # True
```

### Operator Membership

Cek apakah elemen ada di koleksi.

```python
# in dan not in
lst = [1, 2, 3, 4, 5]
print(3 in lst)          # True
print(10 in lst)         # False
print(10 not in lst)     # True

# Untuk string
text = "Hello World"
print("Hello" in text)   # True
print("x" in text)       # False

# Untuk dictionary (cek keys)
d = {"a": 1, "b": 2}
print("a" in d)          # True
print(1 in d)            # False (1 adalah value, bukan key)
```

---

## 🔀 Control Flow: If Statements

### If-Elif-Else

```python
nilai = 85

# Simple if-else
if nilai >= 90:
    print("Grade A")
else:
    print("Grade bukan A")

# If-elif-else
if nilai >= 90:
    grade = "A"
elif nilai >= 80:
    grade = "B"
elif nilai >= 70:
    grade = "C"
else:
    grade = "D"

print(f"Grade: {grade}")
```

### Kondisi Kompleks

```python
umur = 20
punya_sim = True
punya_kendaraan = False

# Multiple conditions dengan and
if umur >= 17 and punya_sim:
    print("Boleh mengendarai")
else:
    print("Tidak boleh")

# Multiple conditions dengan or
if punya_sim or (umur >= 18 and punya_kendaraan):
    print("Kondisi terpenuhi")
else:
    print("Kondisi tidak terpenuhi")

# Nested if
if umur >= 17:
    if punya_sim:
        print("Boleh mengendarai")
    else:
        print("Perlu buat SIM")
else:
    print("Belum cukup umur")
```

### Ternary Operator

Kondisional dalam satu line.

```python
umur = 20

# Panjang
if umur >= 18:
    status = "Dewasa"
else:
    status = "Remaja"

# Ternary operator
status = "Dewasa" if umur >= 18 else "Remaja"
print(status)

# Nested ternary
nilai = 85
grade = "A" if nilai >= 90 else "B" if nilai >= 80 else "C" if nilai >= 70 else "D"
print(grade)
```

---

## 🔁 Loops

### For Loop

For loop iterasi setiap elemen dalam koleksi atau range.

```python
# Iterate list
buah = ["apel", "jeruk", "mangga"]
for item in buah:
    print(item)
# Output: apel, jeruk, mangga

# Iterate dengan index
for i, item in enumerate(buah):
    print(f"{i}: {item}")
# Output: 0: apel, 1: jeruk, 2: mangga

# Iterate dictionary
mahasiswa = {"nama": "Faaza", "umur": 25}
for key, value in mahasiswa.items():
    print(f"{key}: {value}")

# Range loop
for i in range(5):           # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):        # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):    # 0, 2, 4, 6, 8
    print(i)

for i in range(10, 0, -1):   # 10, 9, 8, ..., 1 (descending)
    print(i)
```

### While Loop

While loop berulang selama kondisi True.

```python
# Simple while
counter = 0
while counter < 5:
    print(f"Counter: {counter}")
    counter += 1  # PENTING! Harus increment counter

# Input loop
while True:
    name = input("Siapa nama kamu? (ketik 'quit' untuk keluar): ")
    if name == "quit":
        break
    print(f"Halo {name}!")

# Condition loop
password = ""
while password != "admin":
    password = input("Masukkan password: ")
    if password != "admin":
        print("Password salah")
print("Password benar!")
```

### Break dan Continue

Kontrol flow dalam loop.

```python
# Break - keluar dari loop
for i in range(10):
    if i == 5:
        break  # keluar loop, tidak lanjut ke 6, 7, ...
    print(i)
# Output: 0, 1, 2, 3, 4

# Continue - skip iterasi saat ini
for i in range(10):
    if i % 2 == 0:
        continue  # skip angka genap, lanjut ke berikutnya
    print(i)
# Output: 1, 3, 5, 7, 9

# Kombinasi
for i in range(1, 11):
    if i == 5:
        continue  # skip 5
    if i == 8:
        break     # stop di 8
    print(i)
# Output: 1, 2, 3, 4, 6, 7
```

### Else dengan Loop

Else block dijalankan jika loop selesai normal (tanpa break).

```python
# For-else
for i in range(5):
    print(i)
else:
    print("Loop selesai normal")
# Output: 0, 1, 2, 3, 4, Loop selesai normal

# For-else dengan break
for i in range(5):
    if i == 3:
        break
    print(i)
else:
    print("Loop selesai normal")
# Output: 0, 1, 2 (else TIDAK dijalankan karena ada break)
```

---

## 📝 Praktik: Control Flow Patterns

### Pattern: Sum List

```python
angka = [1, 2, 3, 4, 5]
total = 0

for num in angka:
    total += num

print(f"Total: {total}")  # 15

# Alternative dengan sum()
print(sum(angka))  # 15
```

### Pattern: Find Element

```python
angka = [10, 20, 30, 40, 50]
cari = 30

found = False
for num in angka:
    if num == cari:
        found = True
        break

if found:
    print(f"{cari} ditemukan")
else:
    print(f"{cari} tidak ditemukan")

# Alternative dengan in
if cari in angka:
    print(f"{cari} ditemukan")
```

### Pattern: Validate Input

```python
while True:
    try:
        umur = int(input("Masukkan umur (1-120): "))
        if 1 <= umur <= 120:
            print(f"Umur valid: {umur}")
            break
        else:
            print("Umur harus antara 1-120")
    except ValueError:
        print("Input harus angka!")
```

### Pattern: Filter List

```python
angka = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
genap = []

for num in angka:
    if num % 2 == 0:
        genap.append(num)

print(genap)  # [2, 4, 6, 8, 10]

# Alternative dengan list comprehension (akan dipelajari next page)
genap = [x for x in angka if x % 2 == 0]
```

---

## 📝 Ringkasan Halaman Ini

### Operator Quick Reference

| Tipe | Contoh | Hasil |
| ---- | ------ | ----- |
| Aritmatika | `10 // 3` | `3` |
| Perbandingan | `5 > 3` | `True` |
| Logika | `True and False` | `False` |
| Membership | `3 in [1,2,3]` | `True` |

### Control Flow Keywords

| Keyword | Fungsi |
| ------- | ------ |
| `if` | Kondisional pertama |
| `elif` | Kondisional alternatif |
| `else` | Default |
| `for` | Loop dengan jumlah tertentu |
| `while` | Loop dengan kondisi |
| `break` | Stop loop |
| `continue` | Skip iterasi |

---

## ✏️ Latihan

### Latihan 1: Kondisional

```python
# BMI Calculator
berat = float(input("Berat badan (kg): "))
tinggi = float(input("Tinggi badan (m): "))

bmi = berat / (tinggi ** 2)

if bmi < 18.5:
    kategori = "Underweight"
elif bmi < 25:
    kategori = "Normal"
elif bmi < 30:
    kategori = "Overweight"
else:
    kategori = "Obese"

print(f"BMI: {bmi:.1f} ({kategori})")
```

### Latihan 2: Loop

```python
# Hitung n faktorial
n = int(input("Hitung faktorial berapa? "))
hasil = 1

for i in range(1, n + 1):
    hasil *= i

print(f"{n}! = {hasil}")

# Contoh: 5! = 1 * 2 * 3 * 4 * 5 = 120
```

### Latihan 3: Kombinasi

```python
# Cari bilangan prima
n = int(input("Cari bilangan prima sampai: "))
prima = []

for num in range(2, n + 1):
    is_prime = True
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            is_prime = False
            break
    
    if is_prime:
        prima.append(num)

print(f"Bilangan prima: {prima}")
```

---

## ❓ FAQ

### Q: Perbedaan `break` vs `continue` vs `pass`?

**A:**
- `break` - keluar dari loop seketika
- `continue` - skip iterasi saat ini, lanjut ke berikutnya
- `pass` - tidak lakukan apa-apa (placeholder)

```python
for i in range(5):
    if i == 1:
        continue  # skip 1
    elif i == 3:
        break     # stop di 3
    elif i == 2:
        pass      # lakukan nothing
    print(i)
# Output: 0, 2
```

### Q: Kapan pakai `for` vs `while`?

**A:**
- `for` - tahu berapa iterasi (list, range, etc)
- `while` - tidak tahu berapa iterasi (sampai kondisi terpenuhi)

```python
# For: tahu range
for i in range(10):
    print(i)

# While: tidak tahu sampai kapan
while user_input != "quit":
    user_input = input("Masukkan perintah: ")
```

### Q: Apa `enumerate()`?

**A:** Fungsi yang memberi index dan value sekaligus:
```python
lst = ["a", "b", "c"]

# Tanpa enumerate
for i in range(len(lst)):
    print(f"{i}: {lst[i]}")

# Dengan enumerate (lebih baik)
for i, val in enumerate(lst):
    print(f"{i}: {val}")
```

---

## 🔗 Referensi

- [Python Control Flow](https://docs.python.org/3/tutorial/controlflow.html)
- [Python Operators](https://www.w3schools.com/python/python_operators.asp)
