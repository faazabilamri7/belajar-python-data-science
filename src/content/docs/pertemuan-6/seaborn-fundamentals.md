---
title: Seaborn - Plotting Statistik Dasar
description: Distribution plots, Categorical plots
sidebar:
  order: 4
---

## 🎨 Apa itu Seaborn?

**Seaborn** adalah library visualization dibangun di atas Matplotlib dengan focus pada:
- **Beautiful defaults** - Plot sudah cantik tanpa customization
- **Statistical visualization** - Built-in untuk common statistical plots
- **Pandas integration** - Work directly dengan DataFrames
- **Easy colors & themes** - Nice color palettes & styles

Seaborn perfect untuk **exploratory analysis** - quick visualization untuk understand data.

---

## 🚀 Setup Seaborn

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set theme
sns.set_theme(style="whitegrid")

# Available styles: darkgrid, whitegrid, dark, white, ticks
# sns.set_style("darkgrid")

# Load dataset
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')

print(tips.head())
print(tips.info())
```

---

## 📊 1. Distribution Plots

Distribution plots untuk memvisualisasikan **distribusi dari satu variabel numerik**.

### Histogram Plot (histplot)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic histplot
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', kde=True, ax=ax)
ax.set_title('Histogram dengan KDE')
plt.show()
```

### KDE Plot (Kernel Density Estimate)

```python
# KDE only (smooth distribution)
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=tips, x='total_bill', ax=ax, fill=True, linewidth=2)
ax.set_title('KDE Plot - Smooth Distribution')
plt.show()

# Multiple KDE dengan hue
sns.kdeplot(data=tips, x='total_bill', hue='time', ax=ax, fill=True)
ax.set_title('KDE Plot - By Time')
plt.show()
```

### ECDF Plot (Empirical Cumulative Distribution Function)

```python
# ECDF shows cumulative distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.ecdfplot(data=tips, x='total_bill', hue='sex', ax=ax)
ax.set_title('ECDF Plot - Cumulative Distribution')
plt.show()
```

### Distribution Plot dengan Hue

```python
# Compare distributions across groups
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data=tips, x='total_bill', kde=True, hue='time', ax=ax)
ax.set_title('Distribution - Lunch vs Dinner')
plt.show()
```

---

## 📦 2. Categorical Plots

Categorical plots untuk visualisasi hubungan **antara categorical & numeric variabel**.

### Count Plot

Count plot untuk **count observations di setiap kategori**.

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=tips, x='day', ax=ax, palette='Set2')
ax.set_title('Count Plot - Observations per Day')
ax.set_ylabel('Count')
plt.show()

# With hue (second category)
sns.countplot(data=tips, x='day', hue='sex', ax=ax, palette='Set1')
ax.set_title('Count Plot - By Day & Gender')
plt.show()
```

### Bar Plot

Bar plot untuk **aggregate numeric values** (default = mean).

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=tips, x='day', y='total_bill', ax=ax, palette='husl')
ax.set_title('Bar Plot - Mean Bill by Day')
ax.set_ylabel('Mean Total Bill ($)')
plt.show()

# With hue untuk compare groups
sns.barplot(data=tips, x='day', y='total_bill', hue='sex', ax=ax, palette='Set2')
ax.set_title('Bar Plot - Mean Bill by Day & Gender')
plt.show()
```

### Box Plot

Box plot untuk **show distribution & quartiles**.

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', ax=ax, palette='Set2')
ax.set_title('Box Plot - Bill Distribution by Day')
plt.show()

# With hue
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex', ax=ax, palette='Set1')
ax.set_title('Box Plot - Bill by Day & Gender')
plt.show()
```

### Violin Plot

Violin plot seperti box plot tapi dengan **KDE untuk show distribution shape**.

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=tips, x='day', y='total_bill', ax=ax, palette='husl')
ax.set_title('Violin Plot - Bill Distribution')
plt.show()

# Split violin (compare within same plot)
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True, ax=ax)
ax.set_title('Violin Plot - Split by Gender')
plt.show()
```

### Strip Plot & Swarm Plot

Strip plot untuk **show individual points**.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Strip plot
sns.stripplot(data=tips, x='day', y='total_bill', ax=axes[0], alpha=0.6, size=8)
axes[0].set_title('Strip Plot - Individual Points')

# Swarm plot (avoid overlapping)
sns.swarmplot(data=tips, x='day', y='total_bill', ax=axes[1], size=8)
axes[1].set_title('Swarm Plot - Non-overlapping')

plt.tight_layout()
plt.show()
```

### Combining Multiple Plots

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Combine violin + swarm
sns.violinplot(data=tips, x='day', y='total_bill', ax=ax, palette='Set2', alpha=0.6)
sns.swarmplot(data=tips, x='day', y='total_bill', ax=ax, color='black', size=4)

ax.set_title('Combined: Violin + Swarm')
plt.show()
```

---

## 🎯 3. Categorical Plot dengan Multiple Groups

### Using Palette untuk Color

```python
# Different color palettes
palettes = ['Set1', 'husl', 'coolwarm', 'pastel']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, palette in zip(axes.flatten(), palettes):
    sns.barplot(data=tips, x='day', y='total_bill', hue='sex', 
                ax=ax, palette=palette)
    ax.set_title(f'Palette: {palette}')

plt.tight_layout()
plt.show()
```

### Choosing Aggregation Function

```python
# Default = mean, dapat customize aggregation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Mean
sns.barplot(data=tips, x='day', y='total_bill', ax=axes[0], estimator=np.mean)
axes[0].set_title('Aggregation: Mean')

# Median
sns.barplot(data=tips, x='day', y='total_bill', ax=axes[1], estimator=np.median)
axes[1].set_title('Aggregation: Median')

# Sum
sns.barplot(data=tips, x='day', y='total_bill', ax=axes[2], estimator=np.sum)
axes[2].set_title('Aggregation: Sum')

plt.tight_layout()
plt.show()
```

---

## 🎯 Order & Grouping

### Controlling Order

```python
# Specify order untuk categories
day_order = ['Thurs', 'Fri', 'Sat', 'Sun']

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=tips, x='day', y='total_bill', order=day_order, ax=ax)
ax.set_title('Specified Order')
plt.show()
```

### Multiple Subplots (FacetGrid Intro)

```python
# FacetGrid untuk buat multiple subplots automatically
g = sns.FacetGrid(tips, col='time', height=5)
g.map(sns.histplot, 'total_bill', kde=True)
plt.show()

# FacetGrid dengan row
g = sns.FacetGrid(tips, row='sex', col='time', height=4)
g.map(sns.scatterplot, 'total_bill', 'tip')
plt.show()
```

---

## 📝 Ringkasan - Seaborn Categorical

| Plot Type | Use Case | Best For |
| --------- | -------- | -------- |
| **Count** | Count observations | Frequency distribution |
| **Bar** | Aggregate values | Mean/median comparison |
| **Box** | Distribution by group | Show quartiles & outliers |
| **Violin** | Distribution shape | Smooth distribution view |
| **Strip/Swarm** | Individual points | Examine actual data |

---

## ✏️ Latihan

### Latihan 1: Distribution Exploration

Load `tips` dataset:
1. Create histplot dengan KDE
2. Create KDE plot dengan `hue='time'`
3. Create ECDF plot
4. Compare distributions - apa yang kamu observe?

### Latihan 2: Categorical Analysis

Analyze tips dataset:
1. Count plot untuk 'day' dengan hue='sex'
2. Bar plot untuk mean 'total_bill' by 'day'
3. Box plot untuk 'total_bill' by 'day'
4. Violin plot dengan split=True untuk gender

### Latihan 3: Multiple Groups

1. Create barplot dengan hue untuk compare groups
2. Try different palettes
3. Use different aggregation (mean, median, sum)
4. Specify order untuk categorical axis

### Latihan 4: FacetGrid

Create FacetGrid plots:
1. col='time' untuk buat 2 subplots
2. row='sex', col='time' untuk 2x2 grid
3. Map scatter plot pada FacetGrid
4. Analyze patterns

---

## 🔗 Referensi

- [Seaborn Distribution Plots](https://seaborn.pydata.org/generated/seaborn.histplot.html)
- [Seaborn Categorical Plots](https://seaborn.pydata.org/generated/seaborn.boxplot.html)
- [Seaborn Palettes](https://seaborn.pydata.org/tutorial/color_palettes.html)
