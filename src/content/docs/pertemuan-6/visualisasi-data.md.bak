---
title: Visualisasi Data
description: Membuat visualisasi data yang informatif dan menarik dengan Python
sidebar:
  order: 1
---

## 🎯 Tujuan Pembelajaran

![Data Visualization](https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=400&fit=crop)
_Ilustrasi: Visualisasi membantu mengkomunikasikan insights dari data_

Setelah mempelajari materi ini, kamu akan mampu:

- ✅ Memahami prinsip visualisasi data yang efektif
- ✅ Membuat berbagai jenis chart dengan Matplotlib
- ✅ Membuat visualisasi yang menarik dengan Seaborn
- ✅ Memilih jenis visualisasi yang tepat untuk data

---

## 📊 Mengapa Visualisasi Data Penting?

> "A picture is worth a thousand words"

![Chart Types](https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=300&fit=crop)
_Visualisasi membantu menyampaikan cerita dari data_

Visualisasi data membantu:

- 👁️ **Memahami data** dengan cepat
- 🔍 **Menemukan pola** yang tersembunyi
- 📢 **Mengkomunikasikan** temuan ke stakeholder
- 🎯 **Membuat keputusan** berbasis data

### Contoh: Anscombe's Quartet

![Anscombe's Quartet](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/800px-Anscombe%27s_quartet_3.svg.png)
_4 dataset dengan statistik yang SAMA, tapi visualisasi berbeda - inilah pentingnya visualisasi!_

---

## 🛠️ Library Visualisasi Python

### Instalasi

:::tip[Untuk Pengguna Google Colab]
Semua library visualisasi (matplotlib, seaborn) sudah terinstall di Google Colab. Langsung lanjut ke bagian import!

Buka Google Colab di: [colab.research.google.com](https://colab.research.google.com)
:::

Jika menggunakan laptop/PC lokal:

```bash
pip install matplotlib seaborn plotly
```

### Import Library

Sebelum membuat visualisasi, kita perlu import semua library yang diperlukan. Matplotlib adalah library dasar untuk plotting, Seaborn menyediakan style cantik dan plot statistik, sementara Pandas dan NumPy untuk manipulasi data.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Konfigurasi default
plt.style.use('seaborn-v0_8-whitegrid')  # Style yang bersih
plt.rcParams['figure.figsize'] = (10, 6)  # Ukuran default
plt.rcParams['font.size'] = 12
```

---

## 📈 Matplotlib: Library Dasar

### Struktur Dasar Plot

Setiap plot di Matplotlib memiliki struktur dasar yang sama: membuat figure, plot data, menambahkan judul dan labels, dan menampilkan hasil. Ini adalah template yang akan kita gunakan berulang kali untuk berbagai jenis visualisasi.

```python
import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Membuat plot
plt.figure(figsize=(10, 6))  # Ukuran figure
plt.plot(x, y)               # Plot data
plt.title('Judul Grafik')    # Judul
plt.xlabel('Sumbu X')        # Label X
plt.ylabel('Sumbu Y')        # Label Y
plt.grid(True)               # Grid
plt.show()                   # Tampilkan
```

### Line Plot

Line plot digunakan untuk menampilkan trend atau hubungan antara dua variabel kontinu. Biasanya digunakan untuk time series data. Kita bisa plot multiple lines dalam satu figure dengan memberikan label berbeda pada setiap line.

```python
# Data - generate 100 points dari fungsi sin dan cos
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(12, 5))

# Multiple lines - plot dua fungsi dengan warna dan style berbeda
plt.plot(x, y1, label='Sin(x)', color='blue', linewidth=2, linestyle='-')
plt.plot(x, y2, label='Cos(x)', color='red', linewidth=2, linestyle='--')

plt.title('Fungsi Trigonometri', fontsize=14, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()
```

### Scatter Plot

Scatter plot menampilkan hubungan antara dua variabel numerik. Setiap point mewakili satu observasi. Scatter plot sangat berguna untuk melihat korelasi, outliers, dan pola-pola dalam data.

```python
# Generate data - membuat 100 data points dengan hubungan linear + noise
np.random.seed(42)
x = np.random.randn(100)
y = x * 2 + np.random.randn(100) * 0.5  # y = 2x + noise untuk simulasi korelasi
colors = np.random.rand(100)  # warna random untuk setiap point
sizes = np.random.rand(100) * 200

plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Color Value')
plt.title('Scatter Plot dengan Warna dan Ukuran')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### Bar Chart

Bar chart digunakan untuk membandingkan nilai-nilai dari berbagai kategori. Sangat efektif untuk menampilkan data kategorikal dan memudahkan perbandingan antar kategori. Bar chart bisa vertical (standar) atau horizontal (untuk kategori dengan nama panjang).

```python
# Data - kategorical data dengan nilai numerik
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))

# Vertical bar - batang tegak untuk setiap kategori, mudah untuk perbandingan
plt.subplot(1, 2, 1)
plt.bar(categories, values, color='steelblue', edgecolor='black')
plt.title('Vertical Bar Chart')
plt.xlabel('Kategori')
plt.ylabel('Nilai')

# Horizontal bar
plt.subplot(1, 2, 2)
plt.barh(categories, values, color='coral', edgecolor='black')
plt.title('Horizontal Bar Chart')
plt.xlabel('Nilai')
plt.ylabel('Kategori')

plt.tight_layout()
plt.show()
```

### Grouped & Stacked Bar

Grouped Bar Chart menampilkan multiple bars untuk setiap kategori, memungkinkan perbandingan antar grup. Stacked Bar Chart menumpuk bars untuk menunjukkan komposisi total. Mari kita buat Grouped dan Stacked Bar Chart:

```python
# Data
categories = ['2021', '2022', '2023']
product_a = [100, 150, 200]
product_b = [80, 120, 180]
product_c = [60, 90, 140]

x = np.arange(len(categories))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grouped Bar
axes[0].bar(x - width, product_a, width, label='Product A', color='#2ecc71')
axes[0].bar(x, product_b, width, label='Product B', color='#3498db')
axes[0].bar(x + width, product_c, width, label='Product C', color='#e74c3c')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].set_title('Grouped Bar Chart')
axes[0].legend()

# Stacked Bar
axes[1].bar(categories, product_a, label='Product A', color='#2ecc71')
axes[1].bar(categories, product_b, bottom=product_a, label='Product B', color='#3498db')
bottom_c = [a + b for a, b in zip(product_a, product_b)]
axes[1].bar(categories, product_c, bottom=bottom_c, label='Product C', color='#e74c3c')
axes[1].set_title('Stacked Bar Chart')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Histogram

Histogram menampilkan distribusi dari satu variabel numerik. Kita membagi range data ke dalam bins dan menghitung frekuensi (jumlah data) di setiap bin. Histogram sangat berguna untuk memahami shape distribusi data (normal, skewed, bimodal, dll).

```python
# Generate data dari distribusi normal untuk contoh
np.random.seed(42)
data_normal = np.random.normal(100, 15, 1000)  # mean=100, std=15, 1000 data points

plt.figure(figsize=(12, 5))

# Histogram biasa - menampilkan frekuensi di setiap bin
plt.subplot(1, 2, 1)
plt.hist(data_normal, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.axvline(np.mean(data_normal), color='red', linestyle='--', label=f'Mean: {np.mean(data_normal):.1f}')
plt.legend()

# Histogram dengan KDE (Kernel Density Estimation) - smooth curve untuk visualisasi distribusi
plt.subplot(1, 2, 2)
plt.hist(data_normal, bins=30, density=True, color='steelblue', edgecolor='white', alpha=0.7)
# Add KDE manually - visualisasi smooth dari distribusi
from scipy import stats
x = np.linspace(50, 150, 100)
plt.plot(x, stats.norm.pdf(x, np.mean(data_normal), np.std(data_normal)), 'r-', linewidth=2)
plt.title('Histogram dengan Density Curve')
plt.xlabel('Nilai')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
```

### Pie Chart

Pie Chart adalah visualisasi yang berguna untuk menunjukkan komposisi dan proporsi dari suatu whole (total 100%). Pie Chart paling efektif ketika ada beberapa kategori (tidak lebih dari 5-6) dan saat kita ingin membandingkan bagian dari keseluruhan. Mari kita buat Pie Chart:

```python
# Data
labels = ['Python', 'JavaScript', 'Java', 'C++', 'Others']
sizes = [35, 25, 20, 10, 10]
explode = (0.1, 0, 0, 0, 0)  # Highlight Python
colors = ['#3498db', '#f1c40f', '#e74c3c', '#9b59b6', '#95a5a6']

plt.figure(figsize=(10, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Bahasa Pemrograman Populer 2024')
plt.axis('equal')
plt.show()
```

### Box Plot

Box Plot adalah visualisasi yang sangat berguna untuk melihat distribusi data, median, quartiles, dan outliers dalam satu plot. Box plot memudahkan perbandingan distribusi antar grup. Mari kita buat Box Plot:

```python
# Generate data
np.random.seed(42)
data1 = np.random.normal(100, 10, 200)
data2 = np.random.normal(90, 20, 200)
data3 = np.random.normal(110, 15, 200)

plt.figure(figsize=(10, 6))
bp = plt.boxplot([data1, data2, data3], labels=['Group A', 'Group B', 'Group C'],
                  patch_artist=True)

# Warna
colors = ['#2ecc71', '#3498db', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title('Box Plot Comparison')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()
```

### Subplots

Subplots memungkinkan kita membuat multiple plots dalam satu figure. Ini sangat berguna ketika kita ingin membandingkan visualisasi yang berbeda atau melihat different aspects dari data dalam satu pandangan. Mari kita buat subplot dengan multiple plots:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line
axes[0, 0].plot(np.sin(np.linspace(0, 10, 100)))
axes[0, 0].set_title('Line Plot')

# Plot 2: Scatter
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
axes[0, 1].set_title('Scatter Plot')

# Plot 3: Bar
axes[1, 0].bar(['A', 'B', 'C'], [10, 20, 15])
axes[1, 0].set_title('Bar Chart')

# Plot 4: Histogram
axes[1, 1].hist(np.random.randn(1000), bins=30)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

---

## 🎨 Seaborn: Visualisasi Statistik

Seaborn dibangun di atas Matplotlib dengan:

- Default yang lebih cantik
- Integrasi dengan Pandas DataFrame
- Built-in statistical plots

### Setup Seaborn

Seaborn memiliki banyak style dan color palette yang bisa kita gunakan untuk membuat visualisasi lebih menarik dan profesional. Setup Seaborn memungkinkan kita mengatur tema dan style untuk semua plots. Mari kita setup Seaborn:

```python
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
sns.set_palette("husl")
```

### Distribution Plots

Distribution Plots menunjukkan distribusi data numerik dengan berbagai cara - histogram, KDE (Kernel Density Estimate), atau keduanya. Distribution plots berguna untuk memahami shape, center, dan spread dari data. Mari kita buat distribution plots:

```python
# Load sample data
tips = sns.load_dataset("tips")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram + KDE
sns.histplot(data=tips, x="total_bill", kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histplot')

# KDE only
sns.kdeplot(data=tips, x="total_bill", hue="time", ax=axes[0, 1])
axes[0, 1].set_title('KDE Plot')

# ECDF
sns.ecdfplot(data=tips, x="total_bill", hue="time", ax=axes[1, 0])
axes[1, 0].set_title('ECDF Plot')

# Rug plot
sns.rugplot(data=tips, x="total_bill", ax=axes[1, 1])
axes[1, 1].set_title('Rug Plot')

plt.tight_layout()
plt.show()
```

### Categorical Plots

Categorical Plots menunjukkan hubungan antara variabel categorical dan numerik. Ada berbagai jenis categorical plot (strip, swarm, box, violin) yang berguna untuk exploratory analysis. Mari kita buat categorical plots:

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Count plot
sns.countplot(data=tips, x="day", ax=axes[0, 0])
axes[0, 0].set_title('Count Plot')

# Bar plot (dengan agregasi)
sns.barplot(data=tips, x="day", y="total_bill", ax=axes[0, 1])
axes[0, 1].set_title('Bar Plot (Mean)')

# Box plot
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0, 2])
axes[0, 2].set_title('Box Plot')

# Violin plot
sns.violinplot(data=tips, x="day", y="total_bill", ax=axes[1, 0])
axes[1, 0].set_title('Violin Plot')

# Swarm plot
sns.swarmplot(data=tips, x="day", y="total_bill", ax=axes[1, 1], size=4)
axes[1, 1].set_title('Swarm Plot')

# Strip plot
sns.stripplot(data=tips, x="day", y="total_bill", ax=axes[1, 2], alpha=0.5)
axes[1, 2].set_title('Strip Plot')

plt.tight_layout()
plt.show()
```

### Relational Plots

Relational Plots menunjukkan hubungan antara dua atau lebih variabel numerik. Scatter plot adalah relational plot yang paling common. Relational plots berguna untuk menemukan correlation dan pattern antara variabel. Mari kita buat relational plots:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot dengan hue
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time",
                size="size", sizes=(20, 200), ax=axes[0])
axes[0].set_title('Scatter Plot')

# Line plot
flights = sns.load_dataset("flights")
flights_pivot = flights.pivot(index="month", columns="year", values="passengers")
sns.lineplot(data=flights, x="year", y="passengers", hue="month", ax=axes[1])
axes[1].set_title('Line Plot')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
```

### Heatmap

Heatmap adalah cara yang sangat efektif untuk memvisualisasikan correlation matrix atau data 2D lainnya menggunakan warna. Warna yang lebih gelap/cerah menunjukkan nilai yang lebih tinggi/rendah. Heatmap sangat berguna untuk EDA correlation analysis. Mari kita buat heatmap:

```python
# Correlation heatmap
plt.figure(figsize=(10, 8))

# Numeric columns only
numeric_cols = tips.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```

### Pair Plot

Pair Plot adalah teknik untuk melihat semua pairwise relationships dalam dataset yang berisi multiple variabel numerik. Pair Plot membuat scatter plot untuk setiap pair variabel dan histogram untuk diagonal. Ini sangat berguna untuk EDA awal. Mari kita buat pair plot:

```python
# Pair plot untuk melihat semua hubungan
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species", diag_kind="kde")
plt.suptitle('Iris Dataset Pair Plot', y=1.02)
plt.show()
```

### Joint Plot

Joint Plot menggabungkan scatter plot untuk relationship antara dua variabel dengan histogram untuk masing-masing variabel di margins. Joint plot berguna untuk deep dive analysis hubungan antara two specific variables. Mari kita buat joint plot:

```python
# Joint plot untuk 2 variabel
plt.figure(figsize=(8, 8))
g = sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg",
                  height=8, ratio=3)
g.fig.suptitle('Joint Plot: Total Bill vs Tip', y=1.02)
plt.show()
```

### FacetGrid

FacetGrid adalah teknik untuk membuat multiple plots berdasarkan nilai dari categorical variable. Ini memungkinkan kita membuat side-by-side comparison untuk different groups. FacetGrid sangat powerful untuk exploratory analysis. Mari kita buat FacetGrid:

```python
# Multiple plots berdasarkan kategori
g = sns.FacetGrid(tips, col="time", row="smoker", height=4, aspect=1.2)
g.map(sns.scatterplot, "total_bill", "tip")
g.add_legend()
plt.show()
```

---

## 🎯 Memilih Visualisasi yang Tepat

### Panduan Pemilihan Chart

| Tujuan                       | Jenis Chart              |
| ---------------------------- | ------------------------ |
| **Distribusi 1 variabel**    | Histogram, KDE, Box plot |
| **Perbandingan kategori**    | Bar chart, Count plot    |
| **Hubungan 2 variabel**      | Scatter plot, Line plot  |
| **Komposisi**                | Pie chart, Stacked bar   |
| **Korelasi banyak variabel** | Heatmap, Pair plot       |
| **Trend waktu**              | Line plot                |
| **Distribusi + kategori**    | Violin plot, Box plot    |

### Best Practices

#### 1. Judul yang Jelas

```python
plt.title('Hubungan Antara Jam Belajar dan Nilai Ujian', fontsize=14, fontweight='bold')
```

#### 2. Label yang Informatif

```python
plt.xlabel('Jam Belajar per Minggu')
plt.ylabel('Nilai Ujian (0-100)')
```

#### 3. Warna yang Bermakna

```python
# Gunakan color palette yang konsisten
colors = {'Lulus': '#2ecc71', 'Tidak Lulus': '#e74c3c'}
```

#### 4. Hindari Chart yang Menyesatkan

```python
# Jangan potong sumbu Y kecuali ada alasan kuat
plt.ylim(0, None)  # Mulai dari 0
```

---

## 📝 Template Visualisasi

### Template EDA Visual

Template ini adalah workflow standar untuk melakukan EDA visualization secara systematic. Dengan mengikuti template ini, kita bisa melakukan EDA yang comprehensive untuk dataset apapun. Mari kita lihat template EDA:

```python
def visualize_numeric(df, column):
    """Visualisasi untuk kolom numerik"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram
    sns.histplot(df[column], kde=True, ax=axes[0])
    axes[0].set_title(f'Distribution of {column}')

    # Box plot
    sns.boxplot(y=df[column], ax=axes[1])
    axes[1].set_title(f'Box Plot of {column}')

    # Violin plot
    sns.violinplot(y=df[column], ax=axes[2])
    axes[2].set_title(f'Violin Plot of {column}')

    plt.tight_layout()
    plt.show()

def visualize_categorical(df, column):
    """Visualisasi untuk kolom kategorikal"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Count plot
    sns.countplot(data=df, x=column, ax=axes[0])
    axes[0].set_title(f'Count of {column}')
    axes[0].tick_params(axis='x', rotation=45)

    # Pie chart
    df[column].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
    axes[1].set_title(f'Proportion of {column}')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()
```

### Menyimpan Visualisasi

Setelah membuat visualisasi yang sempurna, kita ingin menyimpannya dalam format high-quality untuk laporan atau presentasi. Matplotlib menyediakan berbagai format (PNG, PDF, SVG) dengan kontrol resolution dan DPI. Mari kita simpan visualisasi:

```python
# Simpan dengan kualitas tinggi
plt.savefig('grafik.png', dpi=300, bbox_inches='tight')
plt.savefig('grafik.pdf', bbox_inches='tight')  # Vector format
plt.savefig('grafik.svg', bbox_inches='tight')  # Scalable
```

---

## 📝 Ringkasan

### Talking Points Hari Ini

| Topik                            | Penjelasan                                           |
| -------------------------------- | ---------------------------------------------------- |
| Matplotlib vs Seaborn            | Matplotlib = dasar & fleksibel, Seaborn = cantik & mudah |
| Jenis Grafik (Bar, Line, Scatter, Boxplot) | Bar = kategori, Line = trend, Scatter = hubungan, Boxplot = distribusi |
| Prinsip Design (Gestalt)         | Proximity, similarity, closure, continuity untuk visualisasi efektif |
| Kustomisasi Plot (Label, Legend, Title) | `plt.xlabel()`, `plt.legend()`, `plt.title()` untuk plot informatif |

### Chart Cheat Sheet

| Data Type           | Chart               |
| ------------------- | ------------------- |
| Distribusi numerik  | Histogram, KDE, Box |
| Kategori            | Bar, Count, Pie     |
| Numerik vs Numerik  | Scatter, Line       |
| Numerik vs Kategori | Box, Violin, Bar    |
| Time Series         | Line                |
| Korelasi            | Heatmap             |

---

## ✏️ Latihan

### Latihan 1: Matplotlib

1. Buat line plot dengan 3 garis berbeda
2. Tambahkan legend, title, dan grid
3. Simpan sebagai file PNG

### Latihan 2: Seaborn

1. Load dataset `tips` dari seaborn
2. Buat visualisasi untuk menjawab: "Apakah ada perbedaan tip antara makan siang dan makan malam?"
3. Buat pair plot untuk melihat semua hubungan

### Latihan 3: EDA Visual

1. Download dataset dari Kaggle
2. Buat minimal 5 visualisasi yang informatif
3. Tulis interpretasi untuk setiap visualisasi

---

## ❓ FAQ (Pertanyaan yang Sering Diajukan)

### Q: Harus pakai Matplotlib atau Seaborn?

**A:**

- **Matplotlib** - Lebih flexible, bisa customize apapun, tapi lebih verbose
- **Seaborn** - Lebih mudah untuk plot statistik, better defaults, tapi less flexible

**Solusi**: Gunakan Seaborn untuk quick exploration, Matplotlib untuk customization.

### Q: Berapa color/font size yang ideal untuk visualisasi?

**A:**

- **Font size**: Min 12pt (biar readable), title bisa lebih besar
- **Colors**: Max 5-6 warna berbeda (lebih banyak = confusing)
- **Lines**: Max 3-4 line dalam satu chart (lebih banyak = berantakan)

**Rule of thumb**: Kalau sulit dibaca di proyektor kecil = terlalu kecil!

### Q: Histogram vs KDE plot, mana yang lebih baik?

**A:**

- **Histogram** - Lebih mudah interpret, tapi tergantung bin size
- **KDE** - Smooth, tapi bisa hide detail

**Best practice**: Gunakan keduanya bersama untuk complete picture!

### Q: Heatmap correlation dengan banyak kolom jadi tidak readable

**A:** Beberapa solusi:

1. Filter hanya korelasi tinggi (> 0.7)
2. Gunakan smaller figure size
3. Rotate labels
4. Gunakan cluster (hierarchical clustering)

```python
# Example: highlight hanya korelasi tinggi
mask = np.abs(corr) < 0.5  # mask korelasi < 0.5
sns.heatmap(corr, mask=mask)
```

### Q: Bagaimana cara buat visualisasi interaktif?

**A:** Gunakan Plotly:

```python
import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
fig.show()
```

Plotly buat chart interaktif yang bisa zoom, hover untuk detail, dll.

### Q: Saya ada outlier, bagaimana di-plot tanpa outliersnya overwhelm chart?

**A:** Beberapa strategi:

1. Gunakan log scale: `plt.yscale('log')`
2. Subset data untuk plot: `df[df['col'] < quantile_99]`
3. Faceted plot: Split ke multiple subplots berdasarkan kategori
4. Gunakan different marker size/alpha untuk outliers

### Q: Bagaimana cara explain chart ke non-technical people?

**A:**

1. **Start dengan headline** - Apa insight utama?
2. **Explain axis** - Apa yang diukur?
3. **Point out pattern** - Tunjuk area penting
4. **Deliver message** - Jangan overwhelm dengan details

**Format**: "Chart ini menunjukkan bahwa [key finding]. Kita bisa lihat [pattern]. Ini berarti [implication]."

---

:::tip[Best Practice]
Visualisasi yang baik adalah visualisasi yang menjawab pertanyaan tanpa perlu banyak explanasi. Kalau harus explain lama = chart kurang jelas!
