---
title: Dasar-Dasar Statistik
description: Konsep statistik fundamental yang menopang algoritma Machine Learning
sidebar:
  order: 1
---

## 🎯 Tujuan Pembelajaran

![Statistics](https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=400&fit=crop)
_Ilustrasi: Statistik adalah bahasa Data Science_

Setelah mempelajari materi ini, kamu akan mampu:

- ✅ Memahami ukuran pemusatan data (mean, median, mode)
- ✅ Memahami ukuran penyebaran data (variance, std, range)
- ✅ Mengenal jenis-jenis distribusi data
- ✅ Memahami konsep probabilitas dasar
- ✅ Menerapkan konsep korelasi dan regresi dasar

---

## 📊 Mengapa Statistik Penting untuk Data Science?

Statistik adalah fondasi dari Data Science dan Machine Learning:

1. **EDA** - Memahami karakteristik data
2. **Feature Engineering** - Transformasi data yang tepat
3. **Model Selection** - Pilih algoritma yang sesuai
4. **Evaluation** - Interpretasi hasil model
5. **Communication** - Presentasi temuan dengan confidence

---

:::tip[Untuk Pengguna Google Colab]
Semua library yang dibutuhkan (numpy, pandas, scipy, matplotlib, seaborn) sudah terinstall di Google Colab. Langsung jalankan kode!

Buka Google Colab di: [colab.research.google.com](https://colab.research.google.com)
:::

## 📈 Ukuran Pemusatan (Central Tendency)

### 1. Mean (Rata-rata)

Mean (rata-rata) adalah ukuran pemusatan yang paling umum digunakan. Mean dihitung dengan menjumlahkan semua nilai dan membaginya dengan jumlah data. Mean berguna untuk data yang berdistribusi normal, tetapi sensitif terhadap outlier (nilai ekstrem).

$$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$$

Mari kita hitung mean menggunakan berbagai metode:

```python
import numpy as np
import pandas as pd

data = [85, 90, 78, 92, 88, 95, 70, 85, 91, 89]

# Menghitung mean
mean_manual = sum(data) / len(data)
mean_numpy = np.mean(data)
mean_pandas = pd.Series(data).mean()

print(f"Mean: {mean_numpy:.2f}")  # 86.30
```

**Kapan menggunakan Mean?**

- ✅ Data berdistribusi normal (simetris)
- ❌ Sensitif terhadap outlier

### 2. Median (Nilai Tengah)

Median adalah nilai yang membagi dataset menjadi dua bagian sama besar - 50% data di bawahnya dan 50% di atasnya. Median lebih robust terhadap outlier dibandingkan mean, sehingga lebih baik digunakan ketika ada nilai ekstrem dalam data:

```python
data = [85, 90, 78, 92, 88, 95, 70, 85, 91, 89]
data_dengan_outlier = [85, 90, 78, 92, 88, 95, 70, 85, 91, 1000]

print(f"Median: {np.median(data):.2f}")  # 88.50
print(f"Mean (dengan outlier): {np.mean(data_dengan_outlier):.2f}")  # 177.40
print(f"Median (dengan outlier): {np.median(data_dengan_outlier):.2f}")  # 88.50
```

**Kapan menggunakan Median?**

- ✅ Data memiliki outlier
- ✅ Data skewed (tidak simetris)
- ✅ Data ordinal

### 3. Mode (Nilai yang Paling Sering Muncul)

Mode adalah nilai yang paling sering muncul dalam dataset. Mode sangat berguna untuk data kategorikal (seperti warna, kategori) maupun data numerik. Mari kita cari mode dari dataset:

```python
from scipy import stats

data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
mode_result = stats.mode(data, keepdims=True)
print(f"Mode: {mode_result.mode[0]}")  # 3

# Untuk kategorikal
kategori = ['A', 'B', 'A', 'C', 'A', 'B']
mode_kat = pd.Series(kategori).mode()[0]
print(f"Mode kategori: {mode_kat}")  # A
```

**Kapan menggunakan Mode?**

- ✅ Data kategorikal
- ✅ Mencari nilai yang paling umum

### Perbandingan Mean, Median, Mode

Perbandingan antara mean, median, dan mode sangat berguna untuk memahami karakteristik distribusi data. Mari kita lihat visualisasinya:

```python
# Visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# Data skewed
data_skewed = np.random.exponential(2, 1000)

plt.figure(figsize=(10, 5))
sns.histplot(data_skewed, kde=True)
plt.axvline(np.mean(data_skewed), color='r', label=f'Mean: {np.mean(data_skewed):.2f}')
plt.axvline(np.median(data_skewed), color='g', label=f'Median: {np.median(data_skewed):.2f}')
plt.legend()
plt.title('Right-Skewed Distribution')
plt.show()
```

---

## 📊 Ukuran Penyebaran (Dispersion)

### 1. Range

Range adalah ukuran penyebaran paling sederhana - hanya selisih antara nilai maksimum dan minimum. Range mudah dihitung tetapi sensitif terhadap outlier:

```python
data = [85, 90, 78, 92, 88, 95, 70, 85, 91, 89]

range_val = max(data) - min(data)
print(f"Range: {range_val}")  # 25
```

### 2. Variance (Varians)

Variance mengukur rata-rata dari kuadrat selisih setiap nilai dengan mean. Variance menunjukkan seberapa menyebar data dari mean - semakin besar variance, semakin spread out datanya. Mari kita hitung variance:

$$\sigma^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n}$$

```python
data = [85, 90, 78, 92, 88, 95, 70, 85, 91, 89]

# Populasi variance
var_pop = np.var(data)
print(f"Variance (population): {var_pop:.2f}")

# Sample variance (n-1)
var_sample = np.var(data, ddof=1)
print(f"Variance (sample): {var_sample:.2f}")
```

### 3. Standard Deviation (Simpangan Baku)

Standard Deviation adalah akar kuadrat dari variance. Standard Deviation lebih mudah diinterpretasi karena satuan dan skala-nya sama dengan data asli. Mari kita hitung Standard Deviation:

$$\sigma = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n}}$$

```python
data = [85, 90, 78, 92, 88, 95, 70, 85, 91, 89]

std = np.std(data, ddof=1)  # sample std
print(f"Standard Deviation: {std:.2f}")
```

### 4. Interquartile Range (IQR)

Interquartile Range adalah range antara kuartil pertama (Q1, 25th percentile) dan kuartil ketiga (Q3, 75th percentile). IQR lebih robust terhadap outlier dibandingkan range karena hanya mengukur middle 50% dari data. Mari kita hitung kuartil dan IQR:

```python
data = [85, 90, 78, 92, 88, 95, 70, 85, 91, 89]

Q1 = np.percentile(data, 25)
Q2 = np.percentile(data, 50)  # median
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

print(f"Q1: {Q1}")
print(f"Q2 (Median): {Q2}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
```

### Interpretasi Standard Deviation

Untuk data berdistribusi normal:

- **68%** data berada dalam ±1 std dari mean
- **95%** data berada dalam ±2 std dari mean
- **99.7%** data berada dalam ±3 std dari mean

```python
np.random.seed(42)
data_normal = np.random.normal(100, 15, 10000)  # mean=100, std=15

mean = np.mean(data_normal)
std = np.std(data_normal)

within_1std = np.sum((data_normal >= mean - std) & (data_normal <= mean + std)) / len(data_normal)
within_2std = np.sum((data_normal >= mean - 2*std) & (data_normal <= mean + 2*std)) / len(data_normal)
within_3std = np.sum((data_normal >= mean - 3*std) & (data_normal <= mean + 3*std)) / len(data_normal)

print(f"Within 1 std: {within_1std*100:.1f}%")  # ~68%
print(f"Within 2 std: {within_2std*100:.1f}%")  # ~95%
print(f"Within 3 std: {within_3std*100:.1f}%")  # ~99.7%
```

---

## 📉 Distribusi Data

### 1. Distribusi Normal (Gaussian)

Distribusi paling umum, berbentuk lonceng simetris.

![Normal Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/800px-Normal_Distribution_PDF.svg.png)
_Distribusi Normal dengan berbagai nilai mean dan standard deviation_

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data normal
np.random.seed(42)
data_normal = np.random.normal(loc=100, scale=15, size=10000)

plt.figure(figsize=(10, 5))
plt.hist(data_normal, bins=50, density=True, alpha=0.7)

# Plot PDF
x = np.linspace(50, 150, 100)
plt.plot(x, stats.norm.pdf(x, 100, 15), 'r-', linewidth=2)
plt.title('Normal Distribution (μ=100, σ=15)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

### 2. Skewness (Kemiringan)

Mengukur asimetri distribusi.

![Skewness](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Negative_and_positive_skew_diagrams_%28English%29.svg/800px-Negative_and_positive_skew_diagrams_%28English%29.svg.png)
_Perbandingan distribusi: Left-skewed, Normal, Right-skewed_

```python
from scipy.stats import skew

data_normal = np.random.normal(0, 1, 1000)
data_right_skewed = np.random.exponential(1, 1000)
data_left_skewed = -np.random.exponential(1, 1000)

print(f"Normal skewness: {skew(data_normal):.3f}")       # ≈ 0
print(f"Right skewness: {skew(data_right_skewed):.3f}")  # > 0
print(f"Left skewness: {skew(data_left_skewed):.3f}")    # < 0
```

| Skewness | Interpretasi                 |
| -------- | ---------------------------- |
| ≈ 0      | Simetris (normal)            |
| > 0      | Right-skewed (ekor ke kanan) |
| < 0      | Left-skewed (ekor ke kiri)   |

### 3. Kurtosis

Mengukur "keruncingan" distribusi.

```python
from scipy.stats import kurtosis

data = np.random.normal(0, 1, 1000)
print(f"Kurtosis: {kurtosis(data):.3f}")
```

| Kurtosis | Interpretasi                |
| -------- | --------------------------- |
| ≈ 0      | Mesokurtic (normal)         |
| > 0      | Leptokurtic (lebih runcing) |
| < 0      | Platykurtic (lebih datar)   |

---

## 🎲 Probabilitas Dasar

### Konsep Dasar

**Probabilitas** adalah ukuran kemungkinan suatu kejadian terjadi, bernilai 0 sampai 1.

$$P(A) = \frac{\text{jumlah kejadian A}}{\text{jumlah semua kemungkinan}}$$

```python
# Contoh: Peluang mendapat angka genap pada dadu
kejadian_genap = [2, 4, 6]
semua_kemungkinan = [1, 2, 3, 4, 5, 6]

probabilitas = len(kejadian_genap) / len(semua_kemungkinan)
print(f"P(genap) = {probabilitas:.2f}")  # 0.50
```

### Aturan Probabilitas

Probabilitas memiliki beberapa aturan dasar yang penting untuk dipahami. Aturan-aturan ini membantu kita menghitung probabilitas event yang lebih complex. Mari kita lihat aturan probabilitas:

```python
# P(A atau B) = P(A) + P(B) - P(A dan B)
# Contoh: P(genap ATAU > 3)
P_genap = 3/6
P_lebih_3 = 3/6
P_genap_dan_lebih_3 = 2/6  # {4, 6}

P_genap_atau_lebih_3 = P_genap + P_lebih_3 - P_genap_dan_lebih_3
print(f"P(genap atau > 3) = {P_genap_atau_lebih_3:.2f}")  # 0.67
```

### Distribusi Probabilitas

Distribusi Probabilitas mendeskripsikan bagaimana probabilitas terdistribusi di antara semua kemungkinan value dari suatu random variable. Normal Distribution adalah yang paling common, tetapi ada berbagai jenis distribusi. Mari kita lihat distribusi probabilitas:

```python
from scipy import stats

# Distribusi Binomial
# Contoh: Peluang mendapat 3 heads dari 5 coin flips
n = 5  # jumlah percobaan
p = 0.5  # probabilitas sukses
k = 3  # jumlah sukses yang diinginkan

prob = stats.binom.pmf(k, n, p)
print(f"P(3 heads dari 5 flips) = {prob:.3f}")  # 0.312
```

---

## 🔗 Korelasi

### Apa itu Korelasi?

Korelasi mengukur hubungan linear antara dua variabel.

### Pearson Correlation Coefficient

$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

Nilai r berada antara -1 dan 1:

- **r = 1**: Korelasi positif sempurna
- **r = 0**: Tidak ada korelasi linear
- **r = -1**: Korelasi negatif sempurna

```python
import numpy as np
from scipy import stats

# Generate data
np.random.seed(42)
x = np.random.randn(100)
y_positive = x * 2 + np.random.randn(100) * 0.5  # korelasi positif
y_negative = -x * 2 + np.random.randn(100) * 0.5  # korelasi negatif
y_none = np.random.randn(100)  # tidak ada korelasi

# Hitung korelasi
r_pos, p_pos = stats.pearsonr(x, y_positive)
r_neg, p_neg = stats.pearsonr(x, y_negative)
r_none, p_none = stats.pearsonr(x, y_none)

print(f"Korelasi positif: r = {r_pos:.3f}")
print(f"Korelasi negatif: r = {r_neg:.3f}")
print(f"Tidak berkorelasi: r = {r_none:.3f}")
```

### Visualisasi Korelasi

Untuk memvisual correlation antara variabel, kita bisa menggunakan berbagai teknik. Scatter plot sangat berguna untuk melihat linear relationship, sedangkan correlation matrix memberikan nilai numeric. Mari kita visualisasi korelasi:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(x, y_positive, alpha=0.5)
axes[0].set_title(f'Korelasi Positif (r={r_pos:.2f})')

axes[1].scatter(x, y_negative, alpha=0.5)
axes[1].set_title(f'Korelasi Negatif (r={r_neg:.2f})')

axes[2].scatter(x, y_none, alpha=0.5)
axes[2].set_title(f'Tidak Berkorelasi (r={r_none:.2f})')

plt.tight_layout()
plt.show()
```

### Correlation Matrix

Correlation Matrix adalah tabel yang menunjukkan correlation coefficient antara setiap pair variabel. Correlation matrix memudahkan kita melihat correlation relationships dalam satu pandangan. Mari kita hitung dan visualisasi correlation matrix:

```python
import pandas as pd
import seaborn as sns

# Buat DataFrame
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100),
})
df['D'] = df['A'] * 2 + np.random.randn(100) * 0.5

# Correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
```

### Korelasi ≠ Kausalitas!

:::caution[Penting!]
Korelasi tinggi **tidak berarti** ada hubungan sebab-akibat!

Contoh: Penjualan es krim berkorelasi dengan kasus tenggelam. Apakah es krim menyebabkan tenggelam? Tidak! Keduanya meningkat di musim panas (confounding variable).
:::

---

## 📐 Z-Score (Standard Score)

Z-score menunjukkan berapa standard deviation suatu nilai dari mean.

$$z = \frac{x - \mu}{\sigma}$$

```python
def calculate_zscore(value, mean, std):
    return (value - mean) / std

# Contoh: Nilai ujian
nilai_ujian = [70, 75, 80, 85, 90, 95, 100]
mean = np.mean(nilai_ujian)
std = np.std(nilai_ujian)

nilai = 95
z = calculate_zscore(nilai, mean, std)
print(f"Nilai {nilai} memiliki z-score: {z:.2f}")

# Menggunakan scipy
from scipy import stats
z_scores = stats.zscore(nilai_ujian)
print(f"Z-scores: {z_scores}")
```

### Interpretasi Z-Score

| Z-Score           | Interpretasi        |
| ----------------- | ------------------- |
| z = 0             | Sama dengan mean    |
| z = 1             | 1 std di atas mean  |
| z = -1            | 1 std di bawah mean |
| z > 3 atau z < -3 | Kemungkinan outlier |

---

## 📝 Statistik Deskriptif dengan Pandas

Pandas menyediakan banyak fungsi berguna untuk menghitung statistik deskriptif dengan cepat. Menggunakan built-in methods dari pandas lebih convenient dan efficient dibanding menghitung manual. Mari kita gunakan pandas untuk statistik deskriptif:

```python
import pandas as pd
import numpy as np

# Buat sample data
np.random.seed(42)
df = pd.DataFrame({
    'nilai': np.random.normal(75, 10, 100),
    'jam_belajar': np.random.uniform(1, 10, 100),
    'kelas': np.random.choice(['A', 'B', 'C'], 100)
})

# Statistik lengkap
print("=== STATISTIK DESKRIPTIF ===")
print(df.describe())

print("\n=== PER KELAS ===")
print(df.groupby('kelas')['nilai'].agg(['mean', 'median', 'std', 'min', 'max']))

print("\n=== KORELASI ===")
print(df[['nilai', 'jam_belajar']].corr())
```

---

## 📝 Ringkasan

| Konsep      | Rumus           | Penggunaan               |
| ----------- | --------------- | ------------------------ |
| Mean        | Σx/n            | Rata-rata, data normal   |
| Median      | Nilai tengah    | Data skewed, ada outlier |
| Mode        | Nilai terbanyak | Data kategorikal         |
| Variance    | Σ(x-μ)²/n       | Ukuran penyebaran        |
| Std Dev     | √Variance       | Penyebaran, satuan sama  |
| Z-Score     | (x-μ)/σ         | Standardisasi            |
| Correlation | Pearson r       | Hubungan 2 variabel      |

---

## ✏️ Latihan

### Latihan 1: Central Tendency

Diberikan data nilai mahasiswa: [78, 82, 90, 85, 88, 72, 95, 88, 79, 91]

1. Hitung mean, median, dan mode
2. Jelaskan ukuran mana yang paling tepat untuk merepresentasikan data ini

### Latihan 2: Dispersion

Dengan data yang sama:

1. Hitung range, variance, dan standard deviation
2. Berapa persen data yang berada dalam ±1 std dari mean?

### Latihan 3: Korelasi

1. Generate 2 dataset yang berkorelasi positif
2. Generate 2 dataset yang tidak berkorelasi
3. Buat scatter plot dan hitung korelasi

### Latihan 4: Z-Score

Jika nilai rata-rata kelas adalah 75 dengan std 10:

1. Berapa z-score untuk nilai 95?
2. Nilai berapa yang memiliki z-score = -2?
3. Apakah nilai 120 bisa dianggap outlier?

---

## ❓ FAQ (Pertanyaan yang Sering Diajukan)

### Q: Kapan harus pakai mean vs median?

**A:**

- **Mean** - Data berdistribusi normal, tidak ada outlier ekstrem
- **Median** - Data skewed atau punya outlier (contoh: salary data, real estate prices)

**Rule of thumb**: Hitung keduanya, kalau hasil jauh berbeda = ada outlier = gunakan median.

### Q: Variance vs Standard Deviation, apa bedanya?

**A:**

- **Variance (σ²)** - Rata-rata kuadrat jarak dari mean, satuan berubah (kuadrat)
- **Std Dev (σ)** - Akar dari variance, satuan sama dengan data original

**Contoh**: Jika data adalah uang (Rp), variance dalam Rp², std dev dalam Rp.

Gunakan **std dev** untuk interpretasi karena satuannya sama!

### Q: Apa itu normal distribution dan kenapa penting?

**A:** Normal distribution (Gaussian) adalah distribusi berbentuk kurva lonceng, simetris di sekitar mean. Penting karena:

1. Banyak fenomena alam mengikuti distribusi normal (tinggi, berat, IQ)
2. Banyak statistik test (t-test, ANOVA) assume data normal
3. Banyak ML algorithms bekerja lebih baik dengan data normal

Cek normalitas dengan histogram, Q-Q plot, atau Shapiro-Wilk test.

### Q: Korelasi itu hubungan sebab-akibat?

**A:** **TIDAK**! Ini common mistake:

- **Korelasi** - ada hubungan antara 2 variabel (mereka bergerak bersama)
- **Causation** - satu variable menyebabkan yang lain

Contoh: Ice cream sales dan drowning deaths berkorelasi tinggi, tapi ice cream tidak menyebabkan drowning (keduanya dipengaruhi musim panas).

**Remember: Correlation ≠ Causation**

### Q: Apa bedanya Pearson, Spearman, dan Kendall correlation?

**A:**

- **Pearson** - Untuk hubungan linear, data continuous & normal
- **Spearman** - Untuk hubungan monotonic (tidak perlu linear), data ordinal atau tidak normal
- **Kendall** - Robust terhadap outlier, cocok untuk data kecil

Gunakan Pearson sebagai default, Spearman jika tidak normal atau ordinal.

### Q: Bagaimana interpret confidence interval?

**A:** Confidence interval 95% artinya: kalau kita repeat experiment 100x, 95 kali confidence interval akan mencakup true population parameter (true mean).

**BUKAN**: Ada 95% chance true mean ada dalam CI ini.

Perbedaan subtle tapi penting!

### Q: Tipe data apa yang butuh statistik apa?

**A:**

- **Numeric (Continuous)**: Mean, Std Dev, Correlation, t-test
- **Categorical**: Mode, Frequency, Chi-square test
- **Ordinal**: Median, Percentile, Spearman correlation, Mann-Whitney test

Memilih test yang salah bisa memberikan hasil yang misleading!

---

:::note[Catatan]
Statistik adalah fondasi dari data science. Jangan cuma hafal rumus, tapi pahami _kapan_ dan _kenapa_ menggunakan masing-masing konsep. Ini akan membuat analisismu lebih robust!
