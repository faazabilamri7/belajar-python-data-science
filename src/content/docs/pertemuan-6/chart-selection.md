---
title: Chart Selection Guide
description: Memilih visualisasi yang tepat untuk data & pertanyaan
sidebar:
  order: 6
---

## 🎯 Memilih Chart yang Tepat

Visualisasi yang baik **menjawab pertanyaan spesifik dengan clear & efficient**. Chart type yang salah membuat insight jadi susah dipahami atau bahkan misleading.

---

## 📊 Univariate Data (1 Variabel)

**Question**: Bagaimana distribusi satu variabel?

### Numeric Data

```
┌─ What to show?
│
├─ Distribution shape → Histogram, KDE, Density Plot
│  └─ seaborn.histplot(), sns.kdeplot()
│
├─ Individual values + distribution → Box plot, Violin plot
│  └─ sns.boxplot(), sns.violinplot()
│
└─ Just frequency → Dot plot, Strip plot
   └─ sns.stripplot()
```

**Kode contoh:**

```python
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('tips')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Histogram
sns.histplot(data=data, x='total_bill', kde=True, ax=axes[0])
axes[0].set_title('Histogram + KDE')

# KDE
sns.kdeplot(data=data, x='total_bill', fill=True, ax=axes[1])
axes[1].set_title('KDE Plot')

# Box plot
sns.boxplot(data=data, y='total_bill', ax=axes[2])
axes[2].set_title('Box Plot')

plt.tight_layout()
plt.show()
```

### Categorical Data

```
┌─ What to show?
│
├─ Count frequency → Bar chart, Count plot
│  └─ sns.countplot(), sns.barplot()
│
├─ Proportions → Pie chart, Donut chart
│  └─ plt.pie()
│
└─ Ordered categories → Horizontal bar chart
   └─ plt.barh()
```

**Kode contoh:**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
sns.countplot(data=data, x='day', ax=axes[0], palette='Set2')
axes[0].set_title('Count Plot')

# Pie chart
day_counts = data['day'].value_counts()
axes[1].pie(day_counts, labels=day_counts.index, autopct='%1.1f%%')
axes[1].set_title('Pie Chart')

plt.tight_layout()
plt.show()
```

---

## 📈 Bivariate Data (2 Variabel)

**Question**: Bagaimana hubungan antara dua variabel?

### Numeric vs Numeric

```
┌─ What to show?
│
├─ Correlation/Relationship → Scatter plot, Regression plot
│  └─ sns.scatterplot(), sns.regplot()
│
├─ Trend over time → Line plot
│  └─ sns.lineplot(), plt.plot()
│
├─ Density of relationship → 2D Density, Hexbin plot
│  └─ sns.kdeplot(data, x, y), plt.hexbin()
│
└─ Strength of relationship → Heatmap (for correlation)
   └─ sns.heatmap() with correlation matrix
```

**Kode contoh:**

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter
sns.scatterplot(data=data, x='total_bill', y='tip', ax=axes[0, 0], alpha=0.6)
axes[0, 0].set_title('Scatter Plot')

# Scatter + Regression
sns.regplot(data=data, x='total_bill', y='tip', ax=axes[0, 1], 
            scatter_kws={'alpha': 0.5})
axes[0, 1].set_title('Scatter + Regression')

# 2D Density
sns.kdeplot(data=data, x='total_bill', y='tip', ax=axes[1, 0], fill=True)
axes[1, 0].set_title('2D Density')

# Hexbin (density dengan hexagons)
axes[1, 1].hexbin(data['total_bill'], data['tip'], gridsize=20, cmap='Blues')
axes[1, 1].set_title('Hexbin Density')
axes[1, 1].set_xlabel('Total Bill')
axes[1, 1].set_ylabel('Tip')

plt.tight_layout()
plt.show()
```

### Numeric vs Categorical

```
┌─ What to show?
│
├─ Distribution by group → Box plot, Violin plot, Strip plot
│  └─ sns.boxplot(), sns.violinplot(), sns.stripplot()
│
├─ Mean/aggregate by group → Bar plot
│  └─ sns.barplot()
│
└─ All individual points + distribution → Swarm + Violin
   └─ sns.swarmplot() + sns.violinplot()
```

**Kode contoh:**

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Box plot
sns.boxplot(data=data, x='day', y='total_bill', ax=axes[0, 0])
axes[0, 0].set_title('Box Plot')

# Violin plot
sns.violinplot(data=data, x='day', y='total_bill', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Violin Plot')

# Bar plot
sns.barplot(data=data, x='day', y='total_bill', ax=axes[1, 0])
axes[1, 0].set_title('Bar Plot (Mean)')

# Swarm + Violin
sns.violinplot(data=data, x='day', y='total_bill', ax=axes[1, 1], 
               alpha=0.6, palette='Set2')
sns.swarmplot(data=data, x='day', y='total_bill', ax=axes[1, 1], 
              color='black', size=3)
axes[1, 1].set_title('Violin + Swarm')

plt.tight_layout()
plt.show()
```

### Categorical vs Categorical

```
┌─ What to show?
│
├─ Frequency of combinations → Heatmap, Mosaic plot
│  └─ Pivot table + sns.heatmap()
│
├─ Grouped counts → Grouped bar plot
│  └─ sns.barplot() or sns.countplot() with hue
│
└─ Cross-tabulation → Stacked bar plot
   └─ plt.bar() with stacking
```

**Kode contoh:**

```python
# Pivot table untuk heatmap
pivot = data.pivot_table(values='total_bill', index='day', columns='time', aggfunc='mean')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Heatmap
sns.heatmap(pivot, annot=True, fmt='.1f', ax=axes[0], cmap='YlGnBu')
axes[0].set_title('Heatmap - Mean Bill')

# Grouped bar plot
sns.barplot(data=data, x='day', y='total_bill', hue='time', ax=axes[1])
axes[1].set_title('Grouped Bar Plot')

plt.tight_layout()
plt.show()
```

---

## 🌟 Multivariate Data (3+ Variabel)

**Question**: Bagaimana hubungan antara 3+ variabel?

### Scatter + Additional Dimensions

```python
# Use color, size, shape untuk encode additional variables
sns.scatterplot(data=data, x='total_bill', y='tip',
                hue='time',           # Color by time
                size='party_size',    # Size by party size  
                style='sex')          # Shape by sex
```

### Faceted Plots (Small Multiples)

```
┌─ What to show?
│
├─ Relationship conditional on third variable → FacetGrid
│  └─ sns.FacetGrid(), sns.lmplot()
│
├─ All pairwise relationships → Pairplot
│  └─ sns.pairplot()
│
└─ Summary across groups → Faceted scatter/box/etc
   └─ sns.FacetGrid().map()
```

**Kode contoh:**

```python
iris = sns.load_dataset('iris')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pairplot (all relationships)
g = sns.pairplot(iris, hue='species', height=2.5)
g.fig.suptitle('Pairplot - All Relationships')

# FacetGrid dengan hue
g = sns.FacetGrid(data, col='time', hue='sex', height=5)
g.map(sns.scatterplot, 'total_bill', 'tip')
g.add_legend()
```

### Correlation Matrix Heatmap

```python
# Perfect untuk visualize banyak relationships sekaligus
corr = iris.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix - All Variables')
plt.show()
```

---

## 🚫 Avoid These Chart Mistakes

### ❌ Pie Charts Dengan Banyak Slices

```python
# BAD - 12 categories pie chart
# GOOD - Use bar chart instead

# Bad:
sizes = [5, 7, 3, 8, 2, 6, 4, 9, 1, 5, 3, 6]
plt.pie(sizes)  # ❌ Hard to read!

# Good:
sns.barplot(y=range(len(sizes)), x=sizes)  # ✓ Easier to compare
```

### ❌ 3D Charts

```python
# 3D charts look fancy tapi misleading
# Stick dengan 2D untuk clarity

# Bad: fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# Good: Use color/size instead untuk 3rd dimension
sns.scatterplot(x=x, y=y, hue=z)
```

### ❌ Dual Y-axes

```python
# Dual axes bisa manipulate perception dari relationship
# Gunakan subplots atau normalize axes instead

# Bad:
# ax2 = ax1.twinx()

# Good: Subplots atau normalize
fig, (ax1, ax2) = plt.subplots(1, 2)
```

### ❌ Misleading Axis Scaling

```python
# Jangan potong axis 0 kecuali ada reason kuat
# Ini bisa exaggerate differences

# Bad - starts at 90:
plt.ylim(90, 100)

# Good - start from 0:
plt.ylim(0, max(values) * 1.1)
```

---

## 📋 Quick Reference - Chart Selection

| Data Type | Goal | Chart Type |
| --------- | ---- | ---------- |
| **1 Numeric** | Distribution | Histogram, KDE, Box plot |
| **1 Numeric** | Trend | Line plot |
| **1 Categorical** | Frequency | Bar chart, Count plot, Pie |
| **2 Numeric** | Relationship | Scatter, Regression line |
| **2 Numeric** | Trend | Line plot |
| **Numeric + Categorical** | Distribution by group | Box, Violin, Strip |
| **Numeric + Categorical** | Mean by group | Bar plot |
| **2 Categorical** | Frequency | Heatmap, Grouped bar |
| **3+ Variables** | All relationships | Pairplot |
| **3+ Variables** | Conditional relationship | FacetGrid |
| **Many Numeric** | Correlation | Heatmap |

---

## 📝 Decision Tree - Pick Your Chart

```
START
  │
  ├─ How many variables? 
  │  ├─ 1 → UNIVARIATE
  │  │  ├─ Numeric? → Histogram, KDE, Box plot
  │  │  └─ Categorical? → Bar chart, Pie chart
  │  │
  │  ├─ 2 → BIVARIATE
  │  │  ├─ Numeric + Numeric? 
  │  │  │  ├─ Relationship? → Scatter plot
  │  │  │  └─ Trend? → Line plot
  │  │  │
  │  │  ├─ Numeric + Categorical? → Box/Violin/Bar plot
  │  │  └─ Categorical + Categorical? → Heatmap/Grouped bar
  │  │
  │  └─ 3+ → MULTIVARIATE
  │     ├─ All relationships? → Pairplot
  │     ├─ Conditional relationships? → FacetGrid
  │     └─ Correlation? → Heatmap
  │
  └─ DONE! Create chart.
```

---

## ✏️ Latihan

### Latihan 1: Dataset Exploration

Pick a dataset & determine:
1. How many variables?
2. Variable types (numeric/categorical)?
3. What question to answer?
4. What chart type best?
5. Create the chart

### Latihan 2: Multi-view Analysis

For the tips dataset, answer:
1. What's the distribution of total_bill?
2. How does tip relate to total_bill?
3. Does time (lunch/dinner) affect tip?
4. Create 4-5 visualizations to answer

### Latihan 3: Presentation

Create 1-page visual report dengan:
- 3-4 related charts
- Clear titles & labels
- Consistent styling
- Key insights highlighted

---

## 🔗 Referensi

- [Edward Tufte - Visual Display of Info](https://www.edwardtufte.com/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples.html)
- [Matplotlib Chart Types](https://matplotlib.org/gallery/index.html)
- [Data Visualization Best Practices](https://www.interaction-design.org/literature/topics/data-visualization)
