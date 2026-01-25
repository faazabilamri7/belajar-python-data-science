---
title: Project & Latihan
description: Mini projects, challenges, dan praktik hands-on
sidebar:
  order: 8
---

## 🎯 Mini Projects

### Project 1: Student Grade Analysis

Analisis data nilai siswa dari 3 mata kuliah dan hitung GPA.

```python
import pandas as pd
import numpy as np

# 1. CREATE DATA
data = {
    'nama': ['Andi', 'Budi', 'Citra', 'Diana', 'Eko', 'Faaza'],
    'matematika': [85, 90, 78, 92, 88, 95],
    'fisika': [88, 85, 82, 90, 86, 92],
    'kimia': [82, 88, 80, 88, 84, 90],
    'jurusan': ['SI', 'TI', 'SI', 'TI', 'SI', 'TI']
}

df = pd.DataFrame(data)

# 2. PROCESS
# Calculate GPA (rata-rata 3 mata kuliah)
df['gpa'] = (df['matematika'] + df['fisika'] + df['kimia']) / 3

# Assign grade
def get_grade(gpa):
    if gpa >= 85:
        return 'A'
    elif gpa >= 75:
        return 'B'
    elif gpa >= 65:
        return 'C'
    else:
        return 'D'

df['grade'] = df['gpa'].apply(get_grade)

# Status kelulusan (GPA >= 70)
df['status'] = df['gpa'].apply(lambda x: 'Lulus' if x >= 70 else 'Tidak Lulus')

# 3. ANALYSIS
print("=== HASIL ANALISIS ===")
print(df.to_string(index=False))

print("\n=== STATISTIK ===")
print(f"Rata-rata GPA: {df['gpa'].mean():.2f}")
print(f"GPA Tertinggi: {df['gpa'].max():.2f} ({df.loc[df['gpa'].idxmax(), 'nama']})")
print(f"GPA Terendah: {df['gpa'].min():.2f} ({df.loc[df['gpa'].idxmin(), 'nama']})")

print("\n=== DISTRIBUSI GRADE ===")
print(df['grade'].value_counts().sort_index())

print("\n=== PER JURUSAN ===")
jurusan_stats = df.groupby('jurusan').agg({
    'gpa': ['mean', 'min', 'max'],
    'status': lambda x: (x == 'Lulus').sum()
})
jurusan_stats.columns = ['GPA Mean', 'GPA Min', 'GPA Max', 'Lulus']
print(jurusan_stats)

# 4. EXPORT
df.to_csv('student_grades.csv', index=False)
print("\n✓ Data exported to student_grades.csv")
```

**Output yang diharapkan:**
```
=== HASIL ANALISIS ===
   nama  matematika  fisika  kimia jurusan    gpa grade status
   Andi          85      88     82      SI  85.00    A  Lulus
   Budi          90      85     88      TI  87.67    A  Lulus
  Citra          78      82     80      SI  80.00    B  Lulus
  Diana          92      90     88      TI  90.00    A  Lulus
    Eko          88      86     84      SI  86.00    A  Lulus
  Faaza          95      92     90      TI  92.33    A  Lulus

=== STATISTIK ===
Rata-rata GPA: 86.83
GPA Tertinggi: 92.33 (Faaza)
GPA Terendah: 80.00 (Citra)
```

---

### Project 2: Sales Data Analysis

Analisis data penjualan dengan multiple categories.

```python
import pandas as pd
import numpy as np

# 1. CREATE DATA
np.random.seed(42)
tanggal = pd.date_range('2024-01-01', periods=100)
data = {
    'tanggal': tanggal,
    'kategori': np.random.choice(['Elektronik', 'Pakaian', 'Makanan'], 100),
    'region': np.random.choice(['Jakarta', 'Bandung', 'Surabaya'], 100),
    'penjualan': np.random.randint(50, 500, 100),
    'biaya': np.random.randint(20, 250, 100)
}

df = pd.DataFrame(data)

# 2. PROCESS
# Calculate profit
df['profit'] = df['penjualan'] - df['biaya']
df['profit_margin'] = (df['profit'] / df['penjualan'] * 100).round(2)

# Month from tanggal
df['bulan'] = df['tanggal'].dt.month
df['minggu'] = df['tanggal'].dt.isocalendar().week

# 3. ANALYSIS
print("=== RINGKASAN KESELURUHAN ===")
print(f"Total Penjualan: Rp {df['penjualan'].sum():,.0f}")
print(f"Total Biaya: Rp {df['biaya'].sum():,.0f}")
print(f"Total Profit: Rp {df['profit'].sum():,.0f}")
print(f"Average Profit Margin: {df['profit_margin'].mean():.2f}%")

print("\n=== PENJUALAN PER KATEGORI ===")
kategori_sales = df.groupby('kategori').agg({
    'penjualan': 'sum',
    'profit': 'sum',
    'profit_margin': 'mean'
}).round(2)
kategori_sales.columns = ['Total Penjualan', 'Total Profit', 'Avg Margin %']
print(kategori_sales)

print("\n=== PENJUALAN PER REGION ===")
region_sales = df.groupby('region').agg({
    'penjualan': ['sum', 'mean'],
    'profit': 'sum'
}).round(2)
region_sales.columns = ['Total Penjualan', 'Rata-rata Per Transaksi', 'Total Profit']
print(region_sales)

print("\n=== TOP 5 TRANSAKSI ===")
top5 = df.nlargest(5, 'profit')[['tanggal', 'kategori', 'region', 'penjualan', 'profit']]
print(top5.to_string(index=False))

print("\n=== PERFORMA PER BULAN ===")
monthly = df.groupby('bulan').agg({
    'penjualan': 'sum',
    'profit': 'sum'
})
monthly.columns = ['Total Penjualan', 'Total Profit']
print(monthly)

# 4. EXPORT
df.to_csv('sales_data.csv', index=False)
kategori_sales.to_csv('kategori_summary.csv')
region_sales.to_csv('region_summary.csv')
print("\n✓ Data exported!")
```

---

### Project 3: Data Cleaning & Preprocessing

Praktik membersihkan data yang "kotor" (realistic scenario).

```python
import pandas as pd
import numpy as np

# 1. CREATE "DIRTY" DATA
data = {
    'nama': ['Andi', 'BUDI', 'citra', 'Diana', 'Eko', '', 'Faaza'],
    'email': ['andi@gmail.com', 'budi@yahoo.com', 'CITRA@GMAIL.COM', 
              'diana@email.co.id', 'eko@domain', None, 'faaza@gmail.com'],
    'umur': [25, 30, '22', 28, 'tidak tahu', 26, 24],
    'gaji': [5000000, 'lima juta', 4000000, 4500000, 3500000, None, 6000000],
    'tanggal_join': ['2024-01-01', '2023-06-15', '2023/02/28', '2024-01-10', 
                     '2023-12-05', '2024-01-20', 'invalid']
}

df = pd.DataFrame(data)

print("=== DATA AWAL (KOTOR) ===")
print(df)
print(f"\nMissing values:\n{df.isnull().sum()}")

# 2. CLEANING

# Clean nama: trim whitespace, standardize case
df['nama'] = df['nama'].str.strip().str.title()
df = df[df['nama'] != '']  # Remove empty names

# Clean email: standardize lowercase, validate format
df['email'] = df['email'].str.lower()
df = df[df['email'].str.contains('@', na=False)]  # Remove invalid emails

# Clean umur: convert to numeric, remove invalid
df['umur'] = pd.to_numeric(df['umur'], errors='coerce')
df = df[df['umur'].between(18, 65)]  # Valid age range

# Clean gaji: remove text, convert to numeric
df['gaji'] = pd.to_numeric(df['gaji'], errors='coerce')
df['gaji'] = df['gaji'].fillna(df['gaji'].median())  # Fill with median

# Clean tanggal_join: convert to datetime
df['tanggal_join'] = pd.to_datetime(df['tanggal_join'], errors='coerce')

print("\n=== SETELAH CLEANING ===")
print(df)
print(f"\nData yang valid: {len(df)} records")
print(f"Data yang di-drop: {7 - len(df)} records")

# 3. SUMMARY STATISTICS
print("\n=== SUMMARY STATISTICS ===")
print(df.describe())

# 4. EXPORT
df.to_csv('cleaned_data.csv', index=False)
print("\n✓ Cleaned data exported!")
```

---

## 🎮 Challenge Problems

### Challenge 1: Student Performance Ranking

```python
# PROBLEM:
# Buat ranking untuk 10 siswa berdasarkan average score
# dari 4 mata kuliah. Include percentile rank.

import pandas as pd
import numpy as np

# Data
np.random.seed(42)
siswa_data = {
    'nama': [f'Siswa_{i}' for i in range(1, 11)],
    'mtk': np.random.randint(60, 100, 10),
    'ing': np.random.randint(60, 100, 10),
    'ipa': np.random.randint(60, 100, 10),
    'ips': np.random.randint(60, 100, 10)
}

df = pd.DataFrame(siswa_data)

# TODO: Your code here
# 1. Calculate average score
# 2. Rank students (1 = highest)
# 3. Calculate percentile rank
# 4. Assign grade (A/B/C/D)
# 5. Print top 5 students

# SOLUTION:
df['rata_rata'] = df[['mtk', 'ing', 'ipa', 'ips']].mean(axis=1)
df['rank'] = df['rata_rata'].rank(ascending=False, method='min').astype(int)
df['percentile'] = df['rata_rata'].rank(pct=True).round(4) * 100
df['grade'] = pd.cut(df['rata_rata'], bins=[0, 70, 80, 90, 100], 
                      labels=['D', 'C', 'B', 'A'])

print(df[['nama', 'rata_rata', 'rank', 'percentile', 'grade']].sort_values('rank'))
```

---

### Challenge 2: Time Series Analysis

```python
# PROBLEM:
# Analisis data penjualan harian selama 30 hari
# Hitung moving average, trend, dan seasonal pattern

import pandas as pd
import numpy as np

# Generate time series data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30)
sales = np.random.normal(1000, 200, 30)
df = pd.DataFrame({'tanggal': dates, 'penjualan': sales})

# TODO: Your code here
# 1. Calculate 7-day moving average
# 2. Calculate trend (increasing/decreasing)
# 3. Find peak sales day
# 4. Calculate growth rate day-over-day

# SOLUTION:
df['ma_7day'] = df['penjualan'].rolling(window=7).mean()
df['growth'] = df['penjualan'].pct_change() * 100
df['hari_minggu'] = df['tanggal'].dt.day_name()

print(df[['tanggal', 'penjualan', 'ma_7day', 'growth']])
print(f"\nPeak sales: {df.loc[df['penjualan'].idxmax()]}")
print(f"Average daily sales: Rp {df['penjualan'].mean():,.0f}")
```

---

### Challenge 3: Data Quality Report

```python
# PROBLEM:
# Buat data quality report yang menunjukkan
# missing values, outliers, dan data distribution

import pandas as pd
import numpy as np

# Create data with issues
df = pd.DataFrame({
    'id': range(1, 101),
    'value1': np.random.normal(100, 20, 100),
    'value2': np.random.normal(50, 10, 100),
    'category': np.random.choice(['A', 'B', 'C', None], 100)
})

# Add some missing values
df.loc[np.random.choice(100, 5, replace=False), 'value1'] = np.nan
df.loc[np.random.choice(100, 3, replace=False), 'value2'] = np.nan

# Add some outliers
df.loc[5, 'value1'] = 500
df.loc[10, 'value2'] = 200

# TODO: Your code here
# 1. Generate missing value report
# 2. Detect outliers (using IQR or z-score)
# 3. Show data distribution summary

# SOLUTION:
print("=== MISSING VALUES REPORT ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_report = pd.DataFrame({
    'Column': missing.index,
    'Missing Count': missing.values,
    'Missing %': missing_pct.values
})
print(missing_report[missing_report['Missing Count'] > 0])

print("\n=== OUTLIERS (IQR METHOD) ===")
for col in ['value1', 'value2']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers detected")

print("\n=== DATA DISTRIBUTION ===")
print(df.describe())
```

---

## 📋 Best Practices Checklist

### Saat Membuat DataFrame

- [ ] Check data types (`.dtypes`, `.info()`)
- [ ] Check shape (`.shape`)
- [ ] Check missing values (`.isnull().sum()`)
- [ ] Use `.head()` dan `.tail()` untuk preview
- [ ] Rename columns to lowercase dengan underscores
- [ ] Set appropriate index jika diperlukan

### Saat Melakukan Agregasi

- [ ] Specify aggregation function yang jelas (sum, mean, count, dll)
- [ ] Group by meaningful columns
- [ ] Reset index jika perlu convert group ke column
- [ ] Validate hasil (cek total masih match)

### Saat Melakukan Merge/Join

- [ ] Verify key columns ada di kedua DataFrame
- [ ] Decide join type (inner/left/right/outer) dengan hati-hati
- [ ] Check for duplicate keys
- [ ] Validate row count setelah merge

### Saat Exporting Data

- [ ] Set `index=False` untuk CSV jika tidak perlu
- [ ] Specify encoding (utf-8 recommended)
- [ ] Double-check file path
- [ ] Test import file setelah export

---

## 📝 Ringkasan Project & Latihan

### Key Techniques Covered

| Teknik | Gunakan Untuk |
| ------ | ------------- |
| GroupBy + Agg | Summary statistics per group |
| Apply | Transform data dengan custom function |
| Merge | Combine data dari multiple sources |
| Filter | Select subset based on conditions |
| Pivot | Reshape data untuk analisis |
| Time Series | Analyze temporal patterns |

---

## ✏️ Latihan Mandiri

### Latihan 1: E-commerce Analysis

Analyze produk, kategori, price, dan rating dari dataset e-commerce.

**Tasks:**
1. Load data dari CSV
2. Clean dan standardize kolom
3. Calculate average rating per kategori
4. Find top 10 best-selling produk
5. Analyze price distribution
6. Export summary ke Excel

### Latihan 2: Financial Report

Analyze pendapatan dan pengeluaran bulanan.

**Tasks:**
1. Create dataset untuk 12 bulan
2. Calculate net income (pendapatan - pengeluaran)
3. Calculate percentage change month-over-month
4. Find best dan worst month
5. Forecast trend untuk bulan depan
6. Create pivot table kategori vs bulan

### Latihan 3: HR Analytics

Analyze employee data untuk department, salary, dan performance.

**Tasks:**
1. Load employee data
2. Calculate salary statistics per department
3. Find high performers
4. Analyze salary vs performance correlation
5. Department turnover rate
6. Generate summary report

---

## 🔗 Resource & Referensi

### Official Documentation
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

### Useful Links
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [NumPy Cheat Sheet](https://datacamp.com/cheat-sheets/numpy-cheat-sheet)
- [Mode Analytics SQL Tutorial](https://mode.com/sql-tutorial/) (untuk SQL joins reference)

### Recommended Datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

---

## 🎓 Next Steps

Setelah menguasai Pertemuan 3 (NumPy & Pandas), lanjut ke:

1. **Pertemuan 4: EDA & Data Cleaning** - Lanjutkan dengan teknik exploratory data analysis dan advanced cleaning
2. **Pertemuan 5: Statistics** - Understand statistical concepts untuk data analysis
3. **Pertemuan 6: Visualization** - Visualisasi data menggunakan Matplotlib & Seaborn

---

:::note[Summary]
Pertemuan 3 merupakan fondasi data science dalam Python. Kuasai NumPy dan Pandas dengan:
- ✅ Memahami konsep, bukan menghafal syntax
- ✅ Praktik dengan real datasets
- ✅ Experiment dengan berbagai operasi
- ✅ Solve actual problems

Mari terus belajar dan praktik! 🚀
:::
