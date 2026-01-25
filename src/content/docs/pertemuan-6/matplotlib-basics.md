---
title: Matplotlib - Plotting Dasar
description: Line plot, scatter plot, bar plot, histogram
sidebar:
  order: 2
---

## 📈 Apa itu Matplotlib?

**Matplotlib** adalah library plotting paling fundamental di Python. Digunakan untuk membuat:
- Static plots (untuk papers, reports)
- Publication-quality visualizations
- Foundation untuk Seaborn & Plotly

"If you can do it in Matplotlib, you can do it anywhere" - karena Matplotlib adalah low-level (penuh kontrol).

---

## 🎨 Struktur Dasar Plot

Setiap Matplotlib plot memiliki struktur yang sama:

```python
import matplotlib.pyplot as plt

# 1. Create figure & axes
fig, ax = plt.subplots(figsize=(10, 6))

# 2. Plot data
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

# 3. Add labels & title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_title('My Plot')

# 4. Show
plt.show()
```

**atau (pyplot style, lebih simple)**

```python
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('My Plot')
plt.show()
```

---

## 📍 1. Line Plot

Line plot untuk menampilkan **trend atau relationship antara 2 variabel kontinuous**. Sangat useful untuk time series data.

### Basic Line Plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Simple Line Plot')
plt.xlabel('X')
plt.ylabel('Sin(X)')
plt.grid(True, alpha=0.3)
plt.show()
```

### Multiple Lines

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * 0.5

plt.figure(figsize=(12, 6))

# Plot multiple lines dengan label
plt.plot(x, y1, label='sin(x)', linewidth=2, color='blue')
plt.plot(x, y2, label='cos(x)', linewidth=2, color='red', linestyle='--')
plt.plot(x, y3, label='0.5*sin(x)', linewidth=2, color='green', linestyle=':')

plt.title('Multiple Lines', fontsize=14, fontweight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend(loc='upper right')  # Add legend
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Line Styles & Markers

```python
x = np.linspace(0, 5, 50)

fig, ax = plt.subplots(figsize=(12, 6))

# Different line styles
styles = ['-', '--', '-.', ':']
colors = ['red', 'blue', 'green', 'orange']

for i, (style, color) in enumerate(zip(styles, colors)):
    y = np.sin(x + i)
    ax.plot(x, y, linestyle=style, color=color, linewidth=2.5, 
            marker='o', markersize=6, label=f'Style: {style}')

ax.set_title('Line Styles & Markers')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Filling Area Under Curve

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot line
ax.plot(x, y, color='blue', linewidth=2, label='sin(x)')

# Fill area
ax.fill_between(x, y, alpha=0.3, color='blue')

ax.set_title('Area Under Curve')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 🔵 2. Scatter Plot

Scatter plot untuk menampilkan **relationship atau correlation antara 2 variabel**. Sangat useful untuk menemukan patterns dan outliers.

### Basic Scatter

```python
import numpy as np

np.random.seed(42)
x = np.random.randn(100)
y = x * 2 + np.random.randn(100) * 0.5

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, s=100, color='steelblue', edgecolors='black')
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.show()
```

### Scatter dengan Color & Size

```python
n_points = 100
x = np.random.randn(n_points)
y = x * 2 + np.random.randn(n_points) * 0.5
colors = np.random.rand(n_points)  # Random values 0-1
sizes = np.random.rand(n_points) * 200  # Sizes 0-200

plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, 
                      cmap='viridis', edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Color Value')
plt.title('Scatter Plot dengan Color & Size')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.show()
```

### Scatter dengan Categories

```python
np.random.seed(42)
n = 50

# Create 3 groups
group1_x = np.random.normal(0, 1, n)
group1_y = group1_x * 1.5 + np.random.normal(0, 0.5, n)

group2_x = np.random.normal(5, 1, n)
group2_y = group2_x * 0.5 + np.random.normal(0, 0.5, n)

group3_x = np.random.normal(2.5, 1, n)
group3_y = np.random.normal(5, 2, n)

plt.figure(figsize=(10, 6))
plt.scatter(group1_x, group1_y, label='Group 1', s=100, alpha=0.6, color='red')
plt.scatter(group2_x, group2_y, label='Group 2', s=100, alpha=0.6, color='blue')
plt.scatter(group3_x, group3_y, label='Group 3', s=100, alpha=0.6, color='green')

plt.title('Scatter Plot - Multiple Groups')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📊 3. Bar Chart

Bar chart untuk **membandingkan values across categories**. Sangat effective untuk categorical data.

### Vertical Bar Chart

```python
categories = ['Python', 'JavaScript', 'Java', 'C++', 'C#']
values = [25, 20, 18, 15, 12]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, values, color='steelblue', edgecolor='black', alpha=0.8)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Programming Languages Popularity', fontsize=14, fontweight='bold')
ax.set_ylabel('Popularity (%)', fontsize=12)
ax.set_ylim(0, max(values) * 1.1)  # Add space for labels
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Horizontal Bar Chart

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Horizontal bars better untuk category names yang panjang
bars = ax.barh(categories, values, color='coral', edgecolor='black', alpha=0.8)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{int(width)}%',
            ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_title('Programming Languages Popularity', fontsize=14, fontweight='bold')
ax.set_xlabel('Popularity (%)', fontsize=12)
ax.set_xlim(0, max(values) * 1.2)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Grouped Bar Chart

```python
years = ['2021', '2022', '2023']
python = [15, 20, 25]
javascript = [18, 19, 20]
java = [20, 18, 18]

x = np.arange(len(years))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width, python, width, label='Python', color='#3498db')
bars2 = ax.bar(x, javascript, width, label='JavaScript', color='#e74c3c')
bars3 = ax.bar(x + width, java, width, label='Java', color='#2ecc71')

ax.set_title('Language Popularity Trend', fontsize=14, fontweight='bold')
ax.set_ylabel('Popularity (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Stacked Bar Chart

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Stacked bars
ax.bar(years, python, label='Python', color='#3498db')
ax.bar(years, javascript, bottom=python, label='JavaScript', color='#e74c3c')

bottom_java = [p + j for p, j in zip(python, javascript)]
ax.bar(years, java, bottom=bottom_java, label='Java', color='#2ecc71')

ax.set_title('Language Popularity - Stacked', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Popularity', fontsize=12)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 📈 4. Histogram

Histogram untuk menampilkan **distribution dari satu variabel numerik**. Kita membagi range data ke bins dan count frequency.

### Basic Histogram

```python
import numpy as np

# Generate data dari normal distribution
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

# Add mean line
mean_val = np.mean(data)
ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')

# Add median line
median_val = np.median(data)
ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')

ax.set_title('Distribution Histogram', fontsize=14, fontweight='bold')
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### Histogram dengan KDE (Kernel Density Estimate)

```python
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(10, 6))

# Histogram
ax.hist(data, bins=30, density=True, color='steelblue', alpha=0.6, edgecolor='black', label='Histogram')

# KDE - smooth curve approximation dari distribution
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 200)
ax.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')

ax.set_title('Histogram dengan KDE', fontsize=14, fontweight='bold')
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### Multiple Histograms (Overlapping)

```python
data1 = np.random.normal(100, 10, 1000)
data2 = np.random.normal(110, 15, 1000)
data3 = np.random.normal(95, 12, 1000)

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(data1, bins=30, alpha=0.5, color='red', label='Group 1')
ax.hist(data2, bins=30, alpha=0.5, color='blue', label='Group 2')
ax.hist(data3, bins=30, alpha=0.5, color='green', label='Group 3')

ax.set_title('Multiple Histograms', fontsize=14, fontweight='bold')
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### Histogram Customization

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Different bin configurations
n, bins, patches = ax.hist(data, bins=50, color='steelblue', edgecolor='white', linewidth=1.5)

# Color gradient untuk bars
cm = plt.cm.viridis
bin_centers = (bins[:-1] + bins[1:]) / 2
col = bin_centers - min(bin_centers)
col /= max(col)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

ax.set_title('Histogram dengan Color Gradient', fontsize=14, fontweight='bold')
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

---

## 📝 Ringkasan - Basic Plots

| Plot Type | Use Case | Best For |
| --------- | -------- | -------- |
| **Line** | Trend, time series | Track changes over time |
| **Scatter** | Relationship, correlation | Find patterns, outliers |
| **Bar** | Category comparison | Compare discrete groups |
| **Histogram** | Distribution | Understand data spread |

---

## ✏️ Latihan

### Latihan 1: Line Plots

Create line plot dengan:
- 2 different lines (sin & cos)
- Different colors, styles, linewidths
- Legend, grid, proper labels
- Title yang descriptive

### Latihan 2: Scatter Analysis

Generate 100 random points dan buat scatter plot untuk show:
- Positive correlation (y = 2x + noise)
- Negative correlation (y = -x + noise)
- No correlation (random y)

Side-by-side comparison.

### Latihan 3: Bar Chart Comparison

Create grouped bar chart dengan:
- 3 categories
- 4 groups per category
- Value labels on bars
- Legend & proper formatting

### Latihan 4: Histogram Exploration

Load real dataset dan create:
- Basic histogram (30 bins)
- Histogram dengan KDE overlay
- Multiple histograms (different groups)
- Analyze distribution shape

---

## 🔗 Referensi

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Matplotlib API Reference](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)
