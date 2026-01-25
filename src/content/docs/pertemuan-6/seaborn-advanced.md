---
title: Seaborn - Advanced Plotting
description: Relational plots, Heatmap, Pairplot, FacetGrid
sidebar:
  order: 5
---

## 🔗 1. Relational Plots

Relational plots untuk visualisasi **hubungan antara 2+ variabel numerik**.

### Scatter Plot (scatterplot)

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# Basic scatter
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip', ax=ax)
ax.set_title('Scatter Plot - Bill vs Tip')
plt.show()
```

### Scatter dengan Hue & Size

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip', 
                hue='time', size='party_size', 
                sizes=(50, 200), alpha=0.6, ax=ax)
ax.set_title('Scatter Plot - dengan Hue & Size')
plt.show()
```

### Scatter dengan Style

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip',
                hue='time', style='sex', 
                markers={'Male': 'o', 'Female': 's'},
                s=200, alpha=0.7, ax=ax)
ax.set_title('Scatter Plot - dengan Style')
plt.show()
```

### Line Plot (lineplot)

```python
# Load dataset dengan time dimension
flights = sns.load_dataset('flights')

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=flights, x='year', y='passengers', hue='month', ax=ax)
ax.set_title('Line Plot - Passengers Over Time')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### Regression Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(data=tips, x='total_bill', y='tip', ax=ax, 
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
ax.set_title('Regression Plot - Linear Fit')
plt.show()

# With categorical
sns.lmplot(data=tips, x='total_bill', y='tip', hue='sex', height=6)
plt.show()
```

---

## 🔥 2. Heatmap

Heatmap untuk visualisasi **2D data sebagai color matrix**. Sangat useful untuk correlation matrices.

### Basic Heatmap

```python
# Calculate correlation matrix
iris = sns.load_dataset('iris')
corr = iris.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
ax.set_title('Correlation Heatmap')
plt.show()
```

### Heatmap dengan Customization

```python
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, 
            annot=True,           # Show values
            fmt='.2f',            # Format: 2 decimals
            cmap='RdBu_r',        # Color palette
            center=0,             # Center at 0
            square=True,          # Square cells
            linewidths=0.5,       # Grid lines
            cbar_kws={'label': 'Correlation'},
            ax=ax)
ax.set_title('Customized Correlation Heatmap', fontsize=14, fontweight='bold')
plt.show()
```

### Pivot Table Heatmap

```python
# Create pivot table
pivot = tips.pivot_table(values='total_bill', index='day', columns='time')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
ax.set_title('Mean Bill - Day vs Time')
plt.show()
```

### Heatmap dengan Masking

```python
# Highlight only high correlations
fig, ax = plt.subplots(figsize=(8, 6))

mask = np.abs(corr) < 0.5  # Mask values < 0.5
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, ax=ax)
ax.set_title('Strong Correlations Only (|r| > 0.5)')
plt.show()
```

### Clustered Heatmap

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Clustered heatmap
g = sns.clustermap(corr, 
                   annot=True, 
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0)
g.fig.suptitle('Clustered Correlation Heatmap', fontsize=14, fontweight='bold')
plt.show()
```

---

## 🎯 3. Pairplot

Pairplot untuk **visualisasi semua pairwise relationships** dalam dataset dengan multiple numeric variables.

### Basic Pairplot

```python
# Pairplot creates scatter matrix
g = sns.pairplot(iris)
plt.suptitle('Iris Dataset Pairplot', y=1.01)
plt.show()
```

### Pairplot dengan Hue

```python
# Color by species
g = sns.pairplot(iris, hue='species', palette='Set2', height=2.5)
g.fig.suptitle('Iris Pairplot - Colored by Species', y=1.00)
plt.show()
```

### Pairplot dengan Diagonal

```python
# Change diagonal plot type
g = sns.pairplot(iris, hue='species', 
                 diag_kind='kde',      # KDE on diagonal
                 plot_kws={'alpha': 0.6},
                 height=2)
g.fig.suptitle('Pairplot - KDE Diagonal', y=1.00)
plt.show()
```

---

## 📊 4. FacetGrid

FacetGrid untuk **membuat multiple subplots otomatis** berdasarkan categorical variable.

### Basic FacetGrid

```python
# Simple FacetGrid
g = sns.FacetGrid(tips, col='time', height=5)
g.map(sns.scatterplot, 'total_bill', 'tip')
g.set_titles('{col_name}')
plt.show()
```

### FacetGrid dengan Row & Col

```python
# 2x2 grid
g = sns.FacetGrid(tips, row='sex', col='time', height=4)
g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.6)
g.set_titles('Sex: {row_name} | Time: {col_name}')
plt.show()
```

### Multiple Geoms pada FacetGrid

```python
# Map multiple plot types
g = sns.FacetGrid(tips, col='time', height=5)
g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.5)
g.map(sns.regplot, 'total_bill', 'tip', scatter=False, color='red')
g.set_titles('Time: {col_name}')
plt.show()
```

### FacetGrid dengan Hue

```python
# FacetGrid dengan additional hue dimension
g = sns.FacetGrid(tips, col='time', hue='sex', height=5, aspect=1.2)
g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.6)
g.add_legend()
plt.show()
```

---

## 🌈 5. Color Palettes

### Available Palettes

```python
# Show all seaborn palettes
palettes = sns.color_palette()
print(sns.color_palette())

# Named palettes
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

named_palettes = ['Set1', 'Set2', 'husl', 'coolwarm', 'dark', 'pastel']

for ax, pal in zip(axes.flatten(), named_palettes):
    sns.barplot(data=tips, x='day', y='total_bill', hue='sex', 
                palette=pal, ax=ax)
    ax.set_title(f'Palette: {pal}')
    ax.set_ylabel('')
    ax.set_xlabel('')

plt.tight_layout()
plt.show()
```

### Custom Palette

```python
# Create custom palette
custom_pal = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=tips, x='day', y='total_bill', hue='sex',
            palette=custom_pal, ax=ax)
ax.set_title('Custom Palette')
plt.show()
```

---

## 🎨 6. Styles & Themes

### Set Style

```python
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, style in zip(axes.flatten(), styles):
    with sns.axes_style(style):
        sns.scatterplot(data=tips, x='total_bill', y='tip', ax=ax)
        ax.set_title(f'Style: {style}')

plt.tight_layout()
plt.show()
```

### Set Context

```python
contexts = ['paper', 'notebook', 'talk', 'poster']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, context in zip(axes.flatten(), contexts):
    with sns.plotting_context(context):
        sns.barplot(data=tips, x='day', y='total_bill', 
                    hue='sex', ax=ax, palette='Set2')
        ax.set_title(f'Context: {context}')

plt.tight_layout()
plt.show()
```

---

## 📝 Ringkasan - Seaborn Advanced

| Plot Type | Use Case | Best For |
| --------- | -------- | -------- |
| **Scatter** | 2 numeric variables | Find relationships |
| **Line** | Time series | Track trends |
| **Regression** | Linear relationships | Show fit line |
| **Heatmap** | 2D data / correlation | Matrix visualization |
| **Pairplot** | All relationships | Multivariate EDA |
| **FacetGrid** | Conditional plots | Compare across groups |

---

## ✏️ Latihan

### Latihan 1: Relational Plots

Using tips dataset:
1. Create scatter plot: total_bill vs tip
2. Add hue=time, size=party_size
3. Create line plot untuk flights data
4. Create regression plot dengan lmplot

### Latihan 2: Correlation Analysis

1. Load iris dataset
2. Calculate correlation matrix
3. Create heatmap dengan annotation
4. Try clustered heatmap
5. Mask low correlations (< 0.5)

### Latihan 3: Pairplot

1. Create pairplot untuk iris data
2. Add hue='species'
3. Change diagonal ke 'kde'
4. Analyze patterns

### Latihan 4: FacetGrid

1. Create FacetGrid dengan col='time'
2. Add row='sex' untuk 2x2 grid
3. Map scatter + regression line
4. Add hue dimension
5. Customize titles

---

## 🔗 Referensi

- [Seaborn Relational Plots](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)
- [Seaborn Heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [Seaborn FacetGrid](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html)
- [Seaborn Styles & Palettes](https://seaborn.pydata.org/tutorial/style_color_defaults.html)
