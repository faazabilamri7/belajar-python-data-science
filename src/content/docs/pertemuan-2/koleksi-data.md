---
title: Koleksi Data
description: List, Dictionary, Tuple dan operasi pada koleksi data
sidebar:
  order: 4
---

## 📊 Koleksi Data (Collections)

### Jenis-jenis Koleksi

| Tipe | Ordered | Mutable | Duplikat | Syntax |
| ---- | ------- | ------- | -------- | ------ |
| List | ✅ Yes | ✅ Yes | ✅ Yes | `[1, 2, 3]` |
| Tuple | ✅ Yes | ❌ No | ✅ Yes | `(1, 2, 3)` |
| Dictionary | ✅ Yes (3.7+) | ✅ Yes | ❌ No (keys) | `{"a": 1}` |
| Set | ❌ No | ✅ Yes | ❌ No | `{1, 2, 3}` |

---

## 📝 List (Array)

### Membuat List

List adalah koleksi data yang **ordered** dan **mutable** (bisa diubah).

```python
# Membuat list
buah = ["apel", "jeruk", "mangga", "pisang"]
angka = [1, 2, 3, 4, 5]
campuran = ["teks", 10, 3.14, True]  # list bisa campur tipe data
kosong = []  # empty list

# List bersarang (nested list)
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### Mengakses Elemen List

```python
buah = ["apel", "jeruk", "mangga", "pisang"]

# Indexing (ambil 1 elemen)
print(buah[0])    # apel - elemen pertama
print(buah[1])    # jeruk - elemen kedua
print(buah[-1])   # pisang - elemen terakhir (index negatif)
print(buah[-2])   # mangga - elemen kedua dari belakang

# Out of range akan error
# print(buah[10])  # Error: IndexError

# Slicing (ambil range)
print(buah[1:3])  # ['jeruk', 'mangga'] - dari index 1 sampai 2
print(buah[:2])   # ['apel', 'jeruk'] - dari awal sampai index 1
print(buah[2:])   # ['mangga', 'pisang'] - dari index 2 sampai akhir
print(buah[::2])  # ['apel', 'mangga'] - setiap 2 elemen
print(buah[::-1]) # ['pisang', 'mangga', 'jeruk', 'apel'] - reverse
```

### Memodifikasi List

```python
buah = ["apel", "jeruk", "mangga"]

# Mengubah elemen
buah[0] = "alpukat"
print(buah)  # ['alpukat', 'jeruk', 'mangga']

# Menambah elemen
buah.append("pisang")  # tambah di akhir
print(buah)  # ['alpukat', 'jeruk', 'mangga', 'pisang']

# Insert di posisi tertentu
buah.insert(1, "semangka")  # masukkan di index 1
print(buah)  # ['alpukat', 'semangka', 'jeruk', 'mangga', 'pisang']

# Menghapus elemen
buah.remove("semangka")  # hapus berdasarkan value
print(buah)  # ['alpukat', 'jeruk', 'mangga', 'pisang']

# Pop elemen (hapus dan return)
item = buah.pop()  # hapus terakhir
print(item)   # pisang
print(buah)   # ['alpukat', 'jeruk', 'mangga']

item = buah.pop(0)  # hapus index 0
print(item)   # alpukat
print(buah)   # ['jeruk', 'mangga']

# Delete statement
del buah[0]
print(buah)  # ['mangga']
```

### List Operations

```python
angka = [1, 2, 3, 4, 5]

# Length
print(len(angka))  # 5

# Check membership
print(3 in angka)      # True
print(10 in angka)     # False
print(3 not in angka)  # False

# Count dan find
print(angka.count(3))  # 1
print(angka.index(3))  # 2 (index dari nilai 3)

# Sort
angka_random = [3, 1, 4, 1, 5]
angka_random.sort()
print(angka_random)  # [1, 1, 3, 4, 5]

angka_random.sort(reverse=True)
print(angka_random)  # [5, 4, 3, 1, 1]

# Reverse
angka.reverse()
print(angka)  # [5, 4, 3, 2, 1]

# Copy
copy_angka = angka.copy()  # atau angka[:]
print(copy_angka)

# Extend (tambah multiple items)
angka.extend([6, 7, 8])
print(angka)  # [5, 4, 3, 2, 1, 6, 7, 8]
```

---

## 🔑 Dictionary (Key-Value Pairs)

### Membuat Dictionary

Dictionary menyimpan data dalam format **key-value pairs** - seperti kamus.

```python
# Membuat dictionary
mahasiswa = {
    "nama": "Faaza",
    "umur": 25,
    "jurusan": "Sistem Informasi",
    "ipk": 3.75
}

# Empty dictionary
kosong = {}
kosong = dict()  # alternative

# Dictionary dengan berbagai tipe value
mixed = {
    "nama": "Faaza",
    "nilai": [85, 90, 88],
    "metadata": {"created": 2024, "updated": 2025}
}
```

### Mengakses Dictionary

```python
mahasiswa = {
    "nama": "Faaza",
    "umur": 25,
    "jurusan": "SI"
}

# Akses dengan key
print(mahasiswa["nama"])    # Faaza
print(mahasiswa["umur"])    # 25

# Akses dengan .get() - lebih aman
print(mahasiswa.get("nama"))  # Faaza
print(mahasiswa.get("alamat"))  # None (tidak error jika key tidak ada)
print(mahasiswa.get("alamat", "N/A"))  # N/A (default value)

# Check key existence
print("nama" in mahasiswa)  # True
print("alamat" in mahasiswa)  # False
```

### Memodifikasi Dictionary

```python
mahasiswa = {
    "nama": "Faaza",
    "umur": 25
}

# Mengubah value
mahasiswa["umur"] = 26
print(mahasiswa)

# Menambah key-value pair baru
mahasiswa["alamat"] = "Jakarta"
mahasiswa["ipk"] = 3.75
print(mahasiswa)

# Menghapus
del mahasiswa["alamat"]
print(mahasiswa)

# Pop dengan default
nilai = mahasiswa.pop("ipk", "N/A")
print(nilai)  # 3.75
print(mahasiswa)
```

### Dictionary Operations

```python
mahasiswa = {
    "nama": "Faaza",
    "umur": 25,
    "jurusan": "SI"
}

# Keys, values, items
print(mahasiswa.keys())      # dict_keys(['nama', 'umur', 'jurusan'])
print(list(mahasiswa.keys()))  # ['nama', 'umur', 'jurusan']

print(mahasiswa.values())    # dict_values(['Faaza', 25, 'SI'])
print(list(mahasiswa.values()))  # ['Faaza', 25, 'SI']

print(mahasiswa.items())     # dict_items([...])
print(list(mahasiswa.items()))  # [('nama', 'Faaza'), ('umur', 25), ...]

# Length
print(len(mahasiswa))  # 3

# Clear
# mahasiswa.clear()  # hapus semua items

# Update
mahasiswa.update({"alamat": "Jakarta", "umur": 26})
print(mahasiswa)

# Copy
copy_mahasiswa = mahasiswa.copy()
```

### Iterasi Dictionary

```python
mahasiswa = {
    "nama": "Faaza",
    "umur": 25,
    "jurusan": "SI"
}

# Iterate keys saja
for key in mahasiswa:
    print(key)  # nama, umur, jurusan

# Iterate values saja
for value in mahasiswa.values():
    print(value)  # Faaza, 25, SI

# Iterate key-value pairs
for key, value in mahasiswa.items():
    print(f"{key}: {value}")
```

---

## (Tuple

### Membuat dan Mengakses Tuple

Tuple mirip list tapi **immutable** (tidak bisa diubah setelah dibuat).

```python
# Membuat tuple
koordinat = (10.5, 20.3)
warna_rgb = (255, 128, 0)
teks = ("hello",)  # single element tuple (perlu koma!)
kosong = ()  # empty tuple

# Mengakses
print(koordinat[0])  # 10.5
print(koordinat[-1])  # 20.3

# Slicing
print(koordinat[:1])  # (10.5,)

# Length
print(len(koordinat))  # 2

# Check membership
print(10.5 in koordinat)  # True
```

### Tuple Unpacking

```python
# Unpacking - assign tuple ke multiple variables
koordinat = (10.5, 20.3)
x, y = koordinat
print(f"x={x}, y={y}")  # x=10.5, y=20.3

# Unpacking dengan multiple values
data = ("Faaza", 25, "Jakarta")
nama, umur, alamat = data
print(f"{nama}, {umur} tahun, tinggal di {alamat}")

# Unpacking dengan _ (ignore)
nama, _, kota = data
print(f"{nama} di {kota}")  # Faaza di Jakarta
```

### Mengapa Tuple?

```python
# Tuple digunakan untuk:
# 1. Data yang tidak boleh berubah
CONSTANT_TUPLE = (1, 2, 3)

# 2. Return multiple values dari function
def get_user_info():
    return ("Faaza", 25, "SI")  # return tuple

nama, umur, jurusan = get_user_info()

# 3. Dictionary keys (list tidak bisa jadi key)
locations = {
    (10, 20): "Jakarta",
    (15, 30): "Bandung"
}
print(locations[(10, 20)])  # Jakarta
```

---

## 🔍 Operasi Umum di Koleksi

### Length

```python
print(len([1, 2, 3]))              # 3
print(len({"a": 1, "b": 2}))       # 2
print(len(("x", "y")))             # 2
```

### Membership

```python
print(2 in [1, 2, 3])              # True
print("a" in {"a": 1, "b": 2})     # True (cek key)
print("x" in ("x", "y"))           # True
```

### Type Conversion

```python
# List ke tuple
list_items = [1, 2, 3]
tuple_items = tuple(list_items)
print(tuple_items)  # (1, 2, 3)

# Tuple ke list
list_back = list(tuple_items)
print(list_back)  # [1, 2, 3]

# Dictionary ke list (keys)
keys = list({"a": 1, "b": 2})
print(keys)  # ['a', 'b']

# Dictionary values to list
values = list({"a": 1, "b": 2}.values())
print(values)  # [1, 2]
```

---

## 📝 Ringkasan Halaman Ini

### Koleksi Data Quick Reference

| Operasi | List | Dict | Tuple |
| ------- | ---- | ---- | ----- |
| Akses | `list[0]` | `dict["key"]` | `tuple[0]` |
| Tambah | `.append()` | `dict["key"] = val` | ❌ Not possible |
| Hapus | `.remove()` | `del dict["key"]` | ❌ Not possible |
| Iterate | `for x in list` | `for k,v in dict.items()` | `for x in tuple` |
| Length | `len(list)` | `len(dict)` | `len(tuple)` |

---

## ✏️ Latihan

### Latihan 1: List Basics

```python
# Buat list nilai ujian 5 mata kuliah
nilai = [85, 90, 78, 92, 88]

# Hitung rata-rata
rata_rata = sum(nilai) / len(nilai)
print(f"Rata-rata: {rata_rata:.2f}")

# Nilai tertinggi dan terendah
print(f"Nilai tertinggi: {max(nilai)}")
print(f"Nilai terendah: {min(nilai)}")

# Tambah nilai baru
nilai.append(95)
print(f"Setelah append: {nilai}")

# Sort
nilai_sorted = sorted(nilai)
print(f"Sorted: {nilai_sorted}")
```

### Latihan 2: Dictionary

```python
# Buat dictionary profile
profile = {
    "nama": "Faaza",
    "umur": 25,
    "hobi": ["coding", "reading"],
    "kontak": {
        "email": "faaza@email.com",
        "phone": "08123456789"
    }
}

# Print profile
for key, value in profile.items():
    print(f"{key}: {value}")

# Akses nested
print(f"\nEmail: {profile['kontak']['email']}")

# Tambah data
profile["alamat"] = "Jakarta"
print(f"\nSetelah tambah: {profile}")
```

### Latihan 3: Kombinasi Koleksi

```python
# List of dictionaries - common pattern!
mahasiswa = [
    {"nama": "Faaza", "jurusan": "SI", "ipk": 3.75},
    {"nama": "Budi", "jurusan": "IF", "ipk": 3.50},
    {"nama": "Citra", "jurusan": "SI", "ipk": 3.80}
]

# Iterate dan print
for mhs in mahasiswa:
    print(f"{mhs['nama']} ({mhs['jurusan']}) - IPK: {mhs['ipk']}")

# Find by name
cari_nama = "Budi"
for mhs in mahasiswa:
    if mhs["nama"] == cari_nama:
        print(f"Ditemukan: {mhs}")
```

---

## ❓ FAQ

### Q: Apa bedanya `remove()` vs `pop()` vs `del`?

**A:**
- `list.remove(value)` - hapus berdasarkan value
- `list.pop(index)` - hapus berdasarkan index, return value
- `del list[index]` - hapus berdasarkan index (statement)

```python
lst = [1, 2, 3, 4]
lst.remove(2)      # [1, 3, 4]
x = lst.pop(0)     # [3, 4], x=1
del lst[0]         # [4]
```

### Q: Bisa pakai dictionary sebagai key dictionary?

**A:** Tidak! Key harus immutable. Gunakan tuple:
```python
# ❌ Error - dictionary tidak immutable
# d = {{1: 2}: "value"}

# ✅ Benar - tuple immutable
d = {(1, 2): "value"}
```

### Q: Apa bedanya `list.copy()` vs assignment `=`?

**A:**
```python
# Assignment - same reference (shallow copy)
a = [1, 2, 3]
b = a
b[0] = 999
print(a)  # [999, 2, 3] - a berubah juga!

# Copy - independent copy
a = [1, 2, 3]
b = a.copy()
b[0] = 999
print(a)  # [1, 2, 3] - a tidak berubah
```

### Q: Perbedaan `dict.get()` vs `dict[key]`?

**A:**
```python
d = {"a": 1}

d["a"]          # 1 - OK
d["b"]          # KeyError!

d.get("a")      # 1 - OK
d.get("b")      # None - tidak error
d.get("b", -1)  # -1 - default value
```

---

## 🔗 Referensi

- [Python Lists Documentation](https://docs.python.org/3/tutorial/datastructures.html)
- [Python Dictionaries](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)
