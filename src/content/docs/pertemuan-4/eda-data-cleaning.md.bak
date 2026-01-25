---
title: EDA & Data Cleaning
description: Teknik mengeksplorasi data dan membersihkan data untuk analisis
sidebar:
  order: 1
---

## 🎯 Tujuan Pembelajaran

![Data Exploration](https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=800&h=400&fit=crop)
_Ilustrasi: EDA membantu memahami data sebelum analisis lanjutan_

Setelah mempelajari materi ini, kamu akan mampu:

- ✅ Melakukan Exploratory Data Analysis (EDA)
- ✅ Mengidentifikasi dan menangani missing values
- ✅ Mendeteksi dan menangani outliers
- ✅ Melakukan data transformation

---

## 🔍 Apa itu EDA?

**Exploratory Data Analysis (EDA)** adalah proses investigasi awal pada data untuk:

- 📊 Menemukan pola dan anomali
- 📈 Memahami distribusi data
- 🔗 Mengidentifikasi hubungan antar variabel
- 🎯 Memformulasikan hipotesis

### EDA dalam CRISP-DM

```
Business Understanding → Data Understanding (EDA!) → Data Preparation → ...
```

---

## 📊 Langkah-langkah EDA

:::tip[Untuk Pengguna Google Colab]
Semua library yang dibutuhkan (pandas, numpy, matplotlib, seaborn) sudah terinstall di Google Colab. Langsung jalankan kode di bawah ini!

Buka Google Colab di: [colab.research.google.com](https://colab.research.google.com)
:::

### 1. Load dan Inspeksi Data

Langkah pertama dalam EDA adalah meload data dan melakukan inspeksi awal. Kita perlu memahami shape dataset (berapa baris dan kolom), tipe data setiap kolom, dan melihat beberapa baris pertama untuk memahami struktur data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data contoh dari seaborn (tidak perlu download!)
df = sns.load_dataset('titanic')  # atau dataset lain

# Atau load dari file CSV:
# df = pd.read_csv('dataset.csv')

# Inspeksi awal
print(f"Shape: {df.shape}")
print(f"\nKolom: {df.columns.tolist()}")
print(f"\nTipe data:\n{df.dtypes}")
print(f"\n5 Data Pertama:\n{df.head()}")
```

### 2. Ringkasan Statistik

Untuk mendapatkan pemahaman yang lebih dalam tentang data kita, kita perlu menghitung statistik deskriptif. Ini memberikan informasi tentang mean, median, standard deviation, dan distribusi data. Mari kita lihat statistik deskriptif:

```python
# Untuk kolom numerik
print(df.describe())

# Untuk kolom kategorikal
print(df.describe(include='object'))

# Info lengkap
print(df.info())
```

### 3. Cek Missing Values

Missing values (data yang hilang atau tidak lengkap) adalah masalah umum dalam dataset real-world. Kita perlu mengidentifikasi ada missing values di mana, berapa banyak, dan dalam persentase berapa. Kita bisa visualisasikan missing values untuk lebih mudah melihat pola-nya. Mari kita identifikasi missing values:

```python
# Jumlah missing values per kolom
print(df.isnull().sum())

# Persentase missing values
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

# Visualisasi missing values
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
```

### 4. Distribusi Data

Memahami distribusi data sangat penting untuk mendeteksi anomali dan outliers. Kita bisa menggunakan histogram untuk melihat distribusi numerik, dan boxplot untuk melihat outliers. Mari kita visualisasikan distribusi data:

```python
# Histogram untuk kolom numerik
df.hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.show()

# Distribusi satu kolom
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
df['kolom'].hist(bins=30)
plt.title('Histogram')

plt.subplot(1, 2, 2)
df['kolom'].plot(kind='box')
plt.title('Boxplot')

plt.tight_layout()
plt.show()
```

### 5. Analisis Kategorikal

Untuk kolom kategorikal (text/categorical), kita perlu melihat value counts (berapa banyak setiap kategori), dan visualisasinya menggunakan bar chart atau pie chart. Ini membantu kita memahami distribusi kategori dan jika ada imbalance. Mari kita analisis kategorikal:

```python
# Value counts
print(df['kategori'].value_counts())

# Visualisasi
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
df['kategori'].value_counts().plot(kind='bar')
plt.title('Bar Chart')

plt.subplot(1, 2, 2)
df['kategori'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Pie Chart')

plt.tight_layout()
plt.show()
```

### 6. Korelasi Antar Variabel

Korelasi menunjukkan hubungan antara dua variabel numerik. Korelasi tinggi (positif atau negatif) menunjukkan hubungan yang kuat, sedangkan korelasi rendah menunjukkan hubungan yang lemah. Kita bisa menggunakan correlation matrix dan heatmap untuk visualisasinya. Mari kita analisis korelasi:

```python
# Correlation matrix
corr_matrix = df.select_dtypes(include=[np.number]).corr()
print(corr_matrix)

# Heatmap korelasi
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

---

## 🧹 Data Cleaning

### Mengapa Data Cleaning Penting?

> "Garbage In, Garbage Out" - Model ML hanya sebaik data yang digunakan

Masalah umum dalam data:

- ❌ Missing values
- ❌ Duplikat
- ❌ Outliers
- ❌ Tipe data salah
- ❌ Inkonsistensi format

---

## 📭 Menangani Missing Values

### Identifikasi Missing Values

Ada berbagai cara data menunjukkan missing values - bisa sebagai 'NA', 'N/A', string kosong, atau angka seperti -1, 999. Kita harus mendefinisikan ini ketika loading data agar pandas bisa mengenali missing values dengan benar:

```python
# Berbagai representasi missing values
missing_indicators = ['NA', 'N/A', '-', '', ' ', 'null', 'None', '?']

# Load dengan definisi missing
df = pd.read_csv('data.csv', na_values=missing_indicators)

# Cek missing
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")
```

### Strategi Menangani Missing Values

Ada berbagai strategi untuk menangani missing values tergantung pada konteks dan persentase missing data. Mari kita lihat berbagai strategi:

#### 1. Hapus Baris/Kolom

Strategi paling sederhana adalah menghapus baris atau kolom yang memiliki missing values. Ini cocok jika jumlah missing tidak terlalu banyak:

```python
# Hapus baris dengan missing values
df_clean = df.dropna()

# Hapus baris jika kolom tertentu missing
df_clean = df.dropna(subset=['kolom_penting'])

# Hapus kolom dengan banyak missing (> 50%)
threshold = 0.5
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df_clean = df.drop(columns=cols_to_drop)
```

#### 2. Imputation (Pengisian)

Daripada menghapus, kita bisa mengisi missing values dengan nilai tertentu. Ada berbagai cara imputation tergantung tipe data dan distribusinya. Mari kita lihat berbagai teknik imputation:

```python
# Isi dengan nilai konstan
df['kolom'].fillna(0, inplace=True)

# Isi dengan mean (untuk numerik)
df['numerik'].fillna(df['numerik'].mean(), inplace=True)

# Isi dengan median (lebih robust terhadap outlier)
df['numerik'].fillna(df['numerik'].median(), inplace=True)

# Isi dengan modus (untuk kategorikal)
df['kategori'].fillna(df['kategori'].mode()[0], inplace=True)

# Forward fill (isi dengan nilai sebelumnya)
df['kolom'].fillna(method='ffill', inplace=True)

# Backward fill (isi dengan nilai setelahnya)
df['kolom'].fillna(method='bfill', inplace=True)
```

#### 3. Imputation per Grup

Untuk data yang terstruktur per grup (misalnya gender, kategori), kita bisa melakukan imputation per grup. Ini lebih akurat karena mempertimbangkan karakteristik grup masing-masing:

```python
# Mean imputation per kategori
df['umur'] = df.groupby('gender')['umur'].transform(
    lambda x: x.fillna(x.mean())
)
```

### Kapan Menggunakan Strategi Mana?

| Situasi                | Strategi              |
| ---------------------- | --------------------- |
| Missing sedikit (< 5%) | Hapus baris           |
| Kolom tidak penting    | Hapus kolom           |
| Data numerik normal    | Mean imputation       |
| Data numerik skewed    | Median imputation     |
| Data kategorikal       | Mode imputation       |
| Time series            | Forward/Backward fill |

---

## 🔄 Menangani Duplikat

Data duplikat (baris yang identik atau hampir identik) dapat menyebabkan bias dalam analisis dan modeling. Kita harus mengidentifikasi dan menghapus duplikat. Mari kita lihat bagaimana mendeteksi dan menghapus duplikat:

```python
# Cek duplikat
print(f"Jumlah duplikat: {df.duplicated().sum()}")

# Lihat baris duplikat
print(df[df.duplicated(keep=False)])

# Hapus duplikat (keep first)
df_clean = df.drop_duplicates()

# Hapus duplikat berdasarkan kolom tertentu
df_clean = df.drop_duplicates(subset=['id', 'tanggal'])

# Hapus duplikat (keep last)
df_clean = df.drop_duplicates(keep='last')
```

---

## 📊 Menangani Outliers

Outlier adalah nilai yang sangat berbeda dari mayoritas data. Outlier bisa disebabkan oleh error pengukuran, data entry error, atau memang ada nilai ekstrem yang legitimate. Kita harus mengidentifikasi outlier dan memutuskan apa yang akan kita lakukan dengannya:

### Apa itu Outlier?

Outlier adalah nilai yang sangat berbeda dari mayoritas data.

### Deteksi Outliers

Mari kita lihat beberapa metode untuk mendeteksi outlier:

#### 1. Metode IQR (Interquartile Range)

```python
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Deteksi
outliers, lb, ub = detect_outliers_iqr(df, 'harga')
print(f"Outliers: {len(outliers)}")
print(f"Range normal: {lb:.2f} - {ub:.2f}")
```

#### 2. Metode Z-Score

Z-Score mengukur berapa banyak standard deviation nilai dari mean. Nilai dengan Z-Score > 3 biasanya dianggap outlier. Mari kita deteksi outlier menggunakan Z-Score:

```python
from scipy import stats

def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers

outliers = detect_outliers_zscore(df, 'harga')
print(f"Outliers: {len(outliers)}")
```

#### 3. Visualisasi Boxplot

Boxplot adalah visualisasi yang sangat berguna untuk mendeteksi outlier secara visual. Titik-titik di luar whisker (garis panjang) pada boxplot adalah outlier. Mari kita visualisasi outlier dengan boxplot:

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
df.boxplot(column='harga')
plt.title('Boxplot - Deteksi Outlier')

plt.subplot(1, 2, 2)
sns.histplot(df['harga'], kde=True)
plt.title('Distribusi Harga')

plt.tight_layout()
plt.show()
```

### Menangani Outliers

Ada beberapa strategi untuk menangani outlier - kita bisa menghapusnya, membatasinya dengan capping, atau mentransformasi mereka. Strategi mana yang kita pilih tergantung pada konteks dan jenis outlier:

```python
# 1. Hapus outliers
Q1 = df['harga'].quantile(0.25)
Q3 = df['harga'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[
    (df['harga'] >= Q1 - 1.5 * IQR) &
    (df['harga'] <= Q3 + 1.5 * IQR)
]

# 2. Capping (Winsorization)
lower = df['harga'].quantile(0.05)
upper = df['harga'].quantile(0.95)
df['harga_capped'] = df['harga'].clip(lower, upper)

# 3. Transformasi (Log)
df['harga_log'] = np.log1p(df['harga'])  # log(1 + x)
```

---

## 🔧 Data Transformation

### Mengubah Tipe Data

Mengubah tipe data adalah langkah penting untuk memastikan pandas memperlakukan kolom dengan benar. Misalnya, kolom tanggal harus bertipe datetime, bukan string. Mari kita ubah tipe data:

```python
# String ke datetime
df['tanggal'] = pd.to_datetime(df['tanggal'])

# Object ke category (hemat memori)
df['kategori'] = df['kategori'].astype('category')

# String ke numerik
df['harga'] = pd.to_numeric(df['harga'], errors='coerce')  # error jadi NaN
```

### Feature Engineering dari Datetime

Datetime columns dapat memberikan informasi tambahan yang berguna untuk modeling. Kita bisa extract year, month, day, day-of-week, quarter dari datetime, dan ini dapat menjadi features yang powerful. Mari kita lakukan feature engineering dari datetime:

```python
df['tanggal'] = pd.to_datetime(df['tanggal'])

# Ekstrak komponen
df['tahun'] = df['tanggal'].dt.year
df['bulan'] = df['tanggal'].dt.month
df['hari'] = df['tanggal'].dt.day
df['hari_dalam_minggu'] = df['tanggal'].dt.dayofweek  # 0=Senin
df['nama_hari'] = df['tanggal'].dt.day_name()
df['kuartal'] = df['tanggal'].dt.quarter
df['is_weekend'] = df['tanggal'].dt.dayofweek >= 5
```

### String Cleaning

Text data sering memiliki inconsistencies seperti extra spaces, inconsistent casing, special characters. String cleaning memastikan text data consistent dan bersih sebelum digunakan untuk modeling. Mari kita clean string data:

```python
# Hapus spasi berlebih
df['nama'] = df['nama'].str.strip()

# Lowercase
df['nama'] = df['nama'].str.lower()

# Ganti karakter
df['telepon'] = df['telepon'].str.replace('-', '')
df['telepon'] = df['telepon'].str.replace(' ', '')

# Ekstrak dengan regex
df['kode_pos'] = df['alamat'].str.extract(r'(\d{5})')
```

### Binning (Kategorisasi Numerik)

Binning adalah teknik untuk mengkonversi continuous numerik variable menjadi categorical variable dengan membaginya ke dalam bins atau ranges. Ini berguna ketika kita ingin menemukan pattern atau ketika model kita lebih cocok dengan categorical input. Mari kita lakukan binning:

```python
# Cut - interval sama
df['umur_grup'] = pd.cut(
    df['umur'],
    bins=[0, 18, 35, 50, 100],
    labels=['Remaja', 'Dewasa Muda', 'Dewasa', 'Lansia']
)

# Qcut - kuantil sama
df['income_quartile'] = pd.qcut(
    df['income'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)
```

---

## 📝 Template EDA

Berikut adalah template atau checklist standar untuk melakukan EDA yang comprehensive. Dengan mengikuti template ini, kita tidak akan melewatkan aspek penting dari data. Mari kita lihat template EDA:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda_report(df):
    """Generate EDA report"""

    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 50)

    # 1. Basic Info
    print("\n1. BASIC INFO")
    print("-" * 30)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 2. Data Types
    print("\n2. DATA TYPES")
    print("-" * 30)
    print(df.dtypes.value_counts())

    # 3. Missing Values
    print("\n3. MISSING VALUES")
    print("-" * 30)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False))

    # 4. Duplicates
    print("\n4. DUPLICATES")
    print("-" * 30)
    print(f"Duplicate Rows: {df.duplicated().sum()}")

    # 5. Numeric Summary
    print("\n5. NUMERIC SUMMARY")
    print("-" * 30)
    print(df.describe())

    # 6. Categorical Summary
    print("\n6. CATEGORICAL SUMMARY")
    print("-" * 30)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head())

    return None

# Gunakan
# eda_report(df)
```

---

## 📊 Visualisasi EDA

Visualisasi adalah bagian krusial dari EDA. Dengan visualisasi yang tepat, kita bisa menemukan pattern, outliers, dan relationships yang tidak terlihat dari angka-angka raw. Mari kita buat berbagai visualisasi untuk EDA:

```python
def plot_eda(df, target=None):
    """Generate EDA visualizations"""

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns

    # 1. Distribution plots
    fig, axes = plt.subplots(len(num_cols), 2, figsize=(12, 4*len(num_cols)))

    for i, col in enumerate(num_cols):
        # Histogram
        axes[i, 0].hist(df[col].dropna(), bins=30, edgecolor='black')
        axes[i, 0].set_title(f'{col} - Histogram')

        # Boxplot
        axes[i, 1].boxplot(df[col].dropna())
        axes[i, 1].set_title(f'{col} - Boxplot')

    plt.tight_layout()
    plt.show()

    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()

# Gunakan
# plot_eda(df)
```

---

## 📝 Ringkasan

### Talking Points Hari Ini

| Topik                      | Penjelasan                                           |
| -------------------------- | ---------------------------------------------------- |
| Identifikasi Missing Values | `df.isnull().sum()`, visualisasi dengan heatmap     |
| Penanganan Outliers        | Deteksi dengan IQR/Z-Score, handling: remove, cap, atau transform |
| Data Type Casting          | `df['col'].astype()`, konversi tipe data yang sesuai |
| Analisis Univariat & Bivariat | Univariat: distribusi 1 variabel, Bivariat: hubungan 2 variabel |

### Checklist EDA & Cleaning

- [ ] Load dan inspeksi data (shape, dtypes, head)
- [ ] Cek dan tangani missing values
- [ ] Deteksi dan tangani outliers
- [ ] Fix data types
- [ ] Analisis univariat dan bivariat
- [ ] Feature engineering

---

## ✏️ Latihan

### Latihan 1: EDA

Download dataset dari Kaggle (misal: Titanic, House Prices), lalu:

1. Lakukan EDA lengkap
2. Identifikasi masalah dalam data
3. Buat visualisasi yang informatif

### Latihan 2: Data Cleaning

1. Tangani semua missing values dengan strategi yang tepat
2. Identifikasi dan tangani outliers
3. Lakukan feature engineering

### Latihan 3: Report

Buat laporan EDA dalam format Jupyter Notebook yang mencakup:

- Deskripsi dataset
- Temuan-temuan penting
- Rekomendasi untuk modeling

---

## ❓ FAQ (Pertanyaan yang Sering Diajukan)

### Q: Berapa persentase missing values yang masih bisa diterima?

**A:** Tergantung context:

- **< 5%** - Aman, bisa di-drop atau di-imput
- **5-20%** - Perlu hati-hati, pertimbangkan imputation method
- **> 20%** - Risky, mungkin ada masalah sistemik pada data collection

Tapi yang paling penting adalah **alasan missing**. Missing karena error recording beda dengan missing by design.

### Q: Harus drop atau impute missing values?

**A:** Tergantung:

- **Drop jika**: Hanya sedikit missing (< 5%), baris random, atau missing tidak informatif
- **Impute jika**: Banyak missing, non-random, atau missing mungkin informatif

Jangan hanya memilih satu - coba keduanya dan bandingkan hasil model!

### Q: Outlier itu berbahaya?

**A:** Outlier tidak selalu buruk:

- **Legitimate outliers** - Data nyata yang ekstrem (contoh: gaji CEO yang sangat tinggi)
- **Error outliers** - Data yang salah karena error recording
- **Influential outliers** - Tidak ekstrem tapi sangat mempengaruhi model

Jangan langsung hapus outlier! Investigasi dulu kenapa ada outlier, baru putuskan.

### Q: Apa itu feature engineering?

**A:** Proses membuat feature (kolom) baru dari feature yang sudah ada untuk meningkatkan model. Contoh:

```python
# Feature baru dari kombinasi
df['age_category'] = df['age'].apply(lambda x: 'muda' if x < 30 else 'tua')

# Feature dari date
df['year'] = pd.to_datetime(df['date']).dt.year
```

### Q: Bagaimana cara tahu kalau data sudah clean?

**A:** Cek beberapa hal:

- ✅ Tidak ada missing values (atau sudah ditangani)
- ✅ Tidak ada duplikat (atau sudah di-remove)
- ✅ Outliers sudah diidentifikasi dan ditangani
- ✅ Tipe data sudah benar
- ✅ Format konsisten (misal date format sama semua)
- ✅ Range nilai reasonable (misal umur bukan -5 atau 500)

### Q: Saya punya data string dengan typo (misal "New York" dan "new york"), bagaimana?

**A:** Normalisasi string:

```python
df['city'] = df['city'].str.lower().str.strip()  # lowercase + hapus spasi
# Atau gunakan fuzzy matching untuk typo yang lebih kompleks
```

### Q: Bagaimana cara handle categorical data dengan banyak categories?

**A:** Beberapa strategi:

1. **Top N categories** - Keep hanya top 10 categories, sisanya "Other"
2. **Grouping** - Combine related categories
3. **Encoding** - Convert ke numerical (one-hot encoding, label encoding)

Pilihan tergantung algoritma yang akan digunakan nanti!

---

:::tip[Pro Tip]
EDA dan Data Cleaning adalah 70-80% dari pekerjaan data scientist! Jangan anggap remeh fase ini. Banyak insight berharga ditemukan di sini, bukan saat modeling.
