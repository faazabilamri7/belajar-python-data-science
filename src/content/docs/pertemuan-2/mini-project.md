---
title: Mini Project & Latihan
description: Praktik dengan project kecil dan challenge problems
sidebar:
  order: 7
---

## 🎯 Mini Project 1: Kalkulator Sederhana

### Requirement

Buat kalkulator yang bisa:
- Tambah, kurang, kali, bagi
- Validasi input
- Handle error

### Solution

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
    str: error message jika ada
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
print(kalkulator(10, 5, "+"))   # 15
print(kalkulator(10, 5, "-"))   # 5
print(kalkulator(10, 5, "*"))   # 50
print(kalkulator(10, 5, "/"))   # 2.0
print(kalkulator(10, 0, "/"))   # Error

# Interactive version
def kalkulator_interaktif():
    """Kalkulator dengan input dari user"""
    while True:
        try:
            a = float(input("Angka pertama: "))
            b = float(input("Angka kedua: "))
            op = input("Operasi (+, -, *, /): ")
            
            hasil = kalkulator(a, b, op)
            print(f"Hasil: {hasil}\n")
            
            lanjut = input("Hitung lagi? (y/n): ")
            if lanjut.lower() != "y":
                break
        except ValueError:
            print("Input harus angka!\n")

# Uncomment untuk run
# kalkulator_interaktif()
```

---

## 🎯 Mini Project 2: Grade Calculator

### Requirement

Hitung grade berdasarkan nilai:
- 90-100: A
- 80-89: B
- 70-79: C
- 60-69: D
- < 60: E

### Solution

```python
def hitung_grade(nilai):
    """Convert nilai menjadi grade"""
    if nilai >= 90:
        return "A"
    elif nilai >= 80:
        return "B"
    elif nilai >= 70:
        return "C"
    elif nilai >= 60:
        return "D"
    else:
        return "E"

def proses_nilai_mahasiswa(data_nilai):
    """
    Process dictionary of students and their grades
    
    data_nilai: dict {nama: nilai, ...}
    """
    hasil = {}
    
    for nama, nilai in data_nilai.items():
        grade = hitung_grade(nilai)
        hasil[nama] = {
            "nilai": nilai,
            "grade": grade
        }
    
    return hasil

# Test
data = {
    "Faaza": 85,
    "Budi": 92,
    "Citra": 78,
    "Doni": 65,
    "Eka": 58
}

hasil = proses_nilai_mahasiswa(data)

# Print dengan formatting
print("=" * 40)
print(f"{'Nama':<15} {'Nilai':<10} {'Grade':<5}")
print("=" * 40)
for nama, info in hasil.items():
    print(f"{nama:<15} {info['nilai']:<10} {info['grade']:<5}")
print("=" * 40)

# Statistik
semua_nilai = [info['nilai'] for info in hasil.values()]
print(f"\nRata-rata: {sum(semua_nilai) / len(semua_nilai):.2f}")
print(f"Nilai tertinggi: {max(semua_nilai)}")
print(f"Nilai terendah: {min(semua_nilai)}")
```

---

## 🎯 Mini Project 3: To-Do List Manager

### Requirement

Manage to-do list dengan:
- Tambah task
- Lihat tasks
- Mark complete
- Delete task

### Solution

```python
class TodoList:
    """Simple to-do list manager"""
    
    def __init__(self):
        self.tasks = []
    
    def add_task(self, task):
        """Add new task"""
        self.tasks.append({
            "task": task,
            "completed": False
        })
        print(f"✅ Task added: {task}")
    
    def view_tasks(self):
        """View all tasks"""
        if not self.tasks:
            print("No tasks yet!")
            return
        
        print("\n" + "=" * 40)
        for i, item in enumerate(self.tasks, 1):
            status = "✓" if item["completed"] else " "
            print(f"{i}. [{status}] {item['task']}")
        print("=" * 40 + "\n")
    
    def complete_task(self, index):
        """Mark task as complete"""
        if 0 < index <= len(self.tasks):
            self.tasks[index - 1]["completed"] = True
            print(f"✅ Task completed!")
        else:
            print("Invalid task number")
    
    def delete_task(self, index):
        """Delete task"""
        if 0 < index <= len(self.tasks):
            removed = self.tasks.pop(index - 1)
            print(f"🗑️ Deleted: {removed['task']}")
        else:
            print("Invalid task number")

# Test
todo = TodoList()
todo.add_task("Belajar Python")
todo.add_task("Buat project")
todo.add_task("Push ke GitHub")

todo.view_tasks()

todo.complete_task(1)
todo.view_tasks()

todo.delete_task(2)
todo.view_tasks()
```

---

## 🏆 Challenge Problems

### Challenge 1: Palindrome Checker

Check apakah string adalah palindrome.

```python
def is_palindrome(text):
    """Check if string is palindrome (ignore spaces and case)"""
    # Remove spaces dan convert to lowercase
    cleaned = text.replace(" ", "").lower()
    # Compare dengan reverse
    return cleaned == cleaned[::-1]

# Test
print(is_palindrome("racecar"))           # True
print(is_palindrome("hello"))             # False
print(is_palindrome("A man a plan a canal Panama"))  # True

# Solution
```

### Challenge 2: Fibonacci Generator

Generate fibonacci sequence sampai n terms.

```python
def fibonacci(n):
    """Generate fibonacci sequence for n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[i-1] + seq[i-2])
    
    return seq

# Test
print(fibonacci(1))   # [0]
print(fibonacci(5))   # [0, 1, 1, 2, 3]
print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### Challenge 3: Count Vowels dan Consonants

```python
def count_letters(text):
    """Count vowels and consonants"""
    vowels = "aeiouAEIOU"
    vowel_count = 0
    consonant_count = 0
    
    for char in text:
        if char.isalpha():  # check if letter
            if char in vowels:
                vowel_count += 1
            else:
                consonant_count += 1
    
    return vowel_count, consonant_count

# Test
text = "Hello World"
vowel, consonant = count_letters(text)
print(f"{text}")
print(f"Vowels: {vowel}, Consonants: {consonant}")

# Alternative dengan list comprehension
def count_letters_v2(text):
    vowels = "aeiouAEIOU"
    letters = [c for c in text if c.isalpha()]
    vowel_count = sum(1 for c in letters if c in vowels)
    consonant_count = len(letters) - vowel_count
    return vowel_count, consonant_count
```

### Challenge 4: Prime Number Checker

```python
def is_prime(n):
    """Check if number is prime"""
    if n < 2:
        return False
    
    # Check dari 2 sampai sqrt(n)
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    
    return True

# Test
print(is_prime(2))      # True
print(is_prime(17))     # True
print(is_prime(4))      # False
print(is_prime(1))      # False

# Find all primes sampai n
def primes_until(n):
    """Find all primes until n"""
    return [x for x in range(2, n) if is_prime(x)]

print(primes_until(20))  # [2, 3, 5, 7, 11, 13, 17, 19]
```

### Challenge 5: Remove Duplicates

```python
def remove_duplicates_v1(lst):
    """Remove duplicates (preserve order using dict)"""
    return list(dict.fromkeys(lst))

def remove_duplicates_v2(lst):
    """Remove duplicates (using set - tidak preserve order)"""
    return list(set(lst))

# Test
data = [1, 2, 2, 3, 3, 3, 4, 5, 5]
print(remove_duplicates_v1(data))  # [1, 2, 3, 4, 5]
print(remove_duplicates_v2(data))  # [1, 2, 3, 4, 5] (order mungkin berbeda)

# Alternative dengan list comprehension
def remove_duplicates_v3(lst):
    """Remove duplicates (manual loop)"""
    result = []
    for item in lst:
        if item not in result:
            result.append(item)
    return result

print(remove_duplicates_v3(data))  # [1, 2, 3, 4, 5]
```

---

## 🎯 Project Ideas untuk Practice

### Idea 1: Temperature Converter

```python
def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9

# Test
print(f"0°C = {celsius_to_fahrenheit(0)}°F")  # 32°F
print(f"100°C = {celsius_to_fahrenheit(100)}°F")  # 212°F
```

### Idea 2: Simple Quiz

```python
def quiz():
    """Simple quiz game"""
    questions = [
        {
            "question": "Berapa 5 + 3?",
            "answer": 8
        },
        {
            "question": "Berapa 10 * 2?",
            "answer": 20
        },
        {
            "question": "Berapa 20 / 4?",
            "answer": 5
        }
    ]
    
    score = 0
    for q in questions:
        print(q["question"])
        jawab = int(input("Jawaban: "))
        
        if jawab == q["answer"]:
            print("✅ Benar!\n")
            score += 1
        else:
            print(f"❌ Salah! Jawaban: {q['answer']}\n")
    
    print(f"Skor akhir: {score}/{len(questions)}")

# Uncomment untuk run
# quiz()
```

### Idea 3: Data Analysis

```python
# Dataset mahasiswa
mahasiswa = [
    {"nama": "Faaza", "nilai": [85, 90, 88]},
    {"nama": "Budi", "nilai": [75, 80, 78]},
    {"nama": "Citra", "nilai": [92, 95, 90]},
    {"nama": "Doni", "nilai": [70, 72, 68]},
]

# Calculate average per student
hasil = []
for mhs in mahasiswa:
    avg = sum(mhs["nilai"]) / len(mhs["nilai"])
    hasil.append({
        "nama": mhs["nama"],
        "rata_rata": avg,
        "grade": "A" if avg >= 85 else "B" if avg >= 75 else "C"
    })

# Print
for r in sorted(hasil, key=lambda x: x["rata_rata"], reverse=True):
    print(f"{r['nama']:<10} {r['rata_rata']:.2f} ({r['grade']})")
```

---

## 📝 Ringkasan

### Skills Practiced

- ✅ Function design dan documentation
- ✅ Input validation
- ✅ Error handling
- ✅ Data structures (lists, dicts)
- ✅ Control flow (loops, conditionals)
- ✅ List comprehension
- ✅ Data processing

### Next Steps

- Continue dengan pertemuan 3 (Pandas & NumPy)
- Practice lebih banyak dengan Kaggle
- Build own mini projects
- Read other people's code

---

## ✏️ Final Latihan

Pilih 1-2 challenge di atas dan complete. Bonus jika bisa:
- Add error handling
- Add input validation
- Make it interactive (ask user for input)
- Optimize dengan list comprehension
- Add documentation/docstrings

**Submit:**
- Run di Google Colab atau local
- Screenshot hasil
- Optional: Share ke GitHub

---

## 🔗 Referensi

- [Python Tutorial - Official](https://docs.python.org/3/tutorial/)
- [GeeksforGeeks - Python](https://www.geeksforgeeks.org/python-programming-language/)
- [Kaggle Learn - Python](https://www.kaggle.com/learn/python)
