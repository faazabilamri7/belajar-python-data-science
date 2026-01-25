---
title: Matplotlib - Advanced Plotting
description: Pie chart, Box plot, Subplots, Customization
sidebar:
  order: 3
---

## 🎨 1. Pie Chart

Pie chart untuk menampilkan **komposisi dan proporsi dari suatu whole**. Effective ketika ada few categories (3-5) dan kita ingin show part-to-whole relationships.

### Basic Pie Chart

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Python', 'JavaScript', 'Java', 'C++', 'Others']
sizes = [35, 25, 20, 10, 10]

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.set_title('Language Distribution 2024', fontsize=14, fontweight='bold')
ax.axis('equal')  # Equal aspect ratio untuk circular pie
plt.tight_layout()
plt.show()
```

### Pie Chart dengan Colors & Explode

```python
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#95a5a6']
explode = (0.1, 0, 0, 0, 0)  # Explode first slice (Python)

fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, explode=explode,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
)

# Customize percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax.set_title('Language Distribution 2024', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Donut Chart (Pie dalam Pie)

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Regular Pie
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.set_title('Pie Chart')
ax1.axis('equal')

# Donut Chart
wedges, texts, autotexts = ax2.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90)
# Gambar circle di tengah untuk jadi donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax2.add_artist(centre_circle)
ax2.set_title('Donut Chart')
ax2.axis('equal')

plt.tight_layout()
plt.show()
```

---

## 📦 2. Box Plot

Box plot untuk menampilkan **distribution dan quartiles dari data**. Sangat useful untuk compare distributions across groups dan identify outliers.

### Basic Box Plot

```python
import numpy as np

np.random.seed(42)
# Generate data dari 3 groups
data1 = np.random.normal(100, 10, 100)
data2 = np.random.normal(95, 20, 100)
data3 = np.random.normal(110, 15, 100)

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([data1, data2, data3], labels=['Group A', 'Group B', 'Group C'],
                  patch_artist=True)

# Customize box colors
colors = ['#3498db', '#e74c3c', '#2ecc71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Value', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Box Plot dengan Detail

```python
fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot([data1, data2, data3], 
                 labels=['Group A', 'Group B', 'Group C'],
                 patch_artist=True,
                 widths=0.6,
                 showmeans=True,  # Show mean dengan marker
                 meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

# Customize
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Customize whiskers & medians
for whisker in bp['whiskers']:
    whisker.set(linewidth=1.5, color='gray')

for median in bp['medians']:
    median.set(linewidth=2, color='darkred')

ax.set_title('Box Plot dengan Mean & Median', fontsize=14, fontweight='bold')
ax.set_ylabel('Value', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 🎯 3. Subplots

Subplots memungkinkan kita membuat multiple plots dalam satu figure. Sangat useful untuk compare visualizations atau show different aspects dari data.

### Basic Subplots (2x2 Grid)

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), color='blue')
axes[0, 0].set_title('Line Plot - Sin(x)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scatter
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50), alpha=0.6)
axes[0, 1].set_title('Scatter Plot')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Bar
axes[1, 0].bar(['A', 'B', 'C', 'D'], [10, 20, 15, 25])
axes[1, 0].set_title('Bar Chart')
axes[1, 0].grid(True, axis='y', alpha=0.3)

# Plot 4: Histogram
axes[1, 1].hist(np.random.normal(100, 15, 1000), bins=30)
axes[1, 1].set_title('Histogram')
axes[1, 1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

### Subplots dengan Different Sizes

```python
# Create grid dengan different sizes
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Large plot spanning 2 columns
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(x, np.sin(x), linewidth=2)
ax1.set_title('Large Plot - Sin(x)')
ax1.grid(True, alpha=0.3)

# Small plot on the right
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(data1, bins=20, alpha=0.7)
ax2.set_title('Histogram')
ax2.grid(True, axis='y', alpha=0.3)

# Bottom row - 3 small plots
for i in range(3):
    ax = fig.add_subplot(gs[1, i])
    ax.scatter(np.random.randn(30), np.random.randn(30), alpha=0.6)
    ax.set_title(f'Scatter {i+1}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Complex Subplot Layout', fontsize=16, fontweight='bold')
plt.show()
```

### Shared Axes

```python
# Create plots dengan shared x-axis
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Different data, same x-axis
axes[0].plot(x, np.sin(x), color='red', linewidth=2)
axes[0].set_ylabel('Sin(x)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, np.cos(x), color='green', linewidth=2)
axes[1].set_ylabel('Cos(x)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(x, np.tan(x), color='blue', linewidth=2)
axes[2].set_ylabel('Tan(x)')
axes[2].set_xlabel('X')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-5, 5)  # Limit tan range

plt.tight_layout()
plt.show()
```

---

## 🎨 4. Customization & Styling

### Color Palettes

```python
# Built-in colormaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

colormaps = ['viridis', 'plasma', 'inferno', 'coolwarm']

for ax, cmap in zip(axes.flatten(), colormaps):
    data = np.random.randn(100)
    colors_data = np.random.rand(100)
    
    scatter = ax.scatter(np.random.randn(100), np.random.randn(100), 
                        c=colors_data, cmap=cmap, s=100, alpha=0.6)
    ax.set_title(f'Colormap: {cmap}')
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()
```

### Fonts & Sizes

```python
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, np.sin(x), linewidth=3, label='sin(x)')
ax.plot(x, np.cos(x), linewidth=3, label='cos(x)')

# Large title
ax.set_title('Customizing Fonts & Sizes', fontsize=18, fontweight='bold', 
             fontfamily='monospace')

# Larger labels
ax.set_xlabel('X Axis', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Axis', fontsize=14, fontweight='bold')

# Larger legend
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

# Larger tick labels
ax.tick_params(axis='both', which='major', labelsize=11)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Annotations & Arrows

```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, linewidth=2)

# Annotate maximum point
max_idx = np.argmax(y)
max_x, max_y = x[max_idx], y[max_idx]
ax.annotate('Maximum', xy=(max_x, max_y), xytext=(max_x+1, max_y+0.5),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Annotate minimum point
min_idx = np.argmin(y)
min_x, min_y = x[min_idx], y[min_idx]
ax.annotate('Minimum', xy=(min_x, min_y), xytext=(min_x+1, min_y-0.5),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))

# Add text box
textstr = 'This is a text box\nwith multiple lines'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

ax.set_title('Annotations & Arrows', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Legends & Positioning

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax in axes:
    ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
    ax.plot(x, np.cos(x), label='cos(x)', linewidth=2)
    ax.plot(x, np.tan(x), label='tan(x)', linewidth=2)
    ax.grid(True, alpha=0.3)

# Different legend positions
axes[0].legend(loc='upper left')
axes[0].set_title('Legend - Upper Left')

axes[1].legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.1))
axes[1].set_title('Legend - Upper Center (Below)')

plt.tight_layout()
plt.show()
```

---

## 💾 Saving Figures

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x))
ax.set_title('My Plot')

# Save dengan different formats & resolutions
plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # PNG format
plt.savefig('plot.pdf', bbox_inches='tight')  # PDF (vector)
plt.savefig('plot.svg', bbox_inches='tight')  # SVG (scalable)

# Transparent background
plt.savefig('plot_transparent.png', dpi=300, bbox_inches='tight', transparent=True)
```

---

## 📝 Ringkasan - Advanced Plots

| Plot Type | Use Case | Notes |
| --------- | -------- | ----- |
| **Pie** | Part-to-whole | Max 5-6 categories |
| **Box** | Distribution comparison | Show outliers & quartiles |
| **Subplots** | Multiple comparisons | Organize many plots |
| **Customization** | Publication quality | Fonts, colors, annotations |

---

## ✏️ Latihan

### Latihan 1: Pie & Donut Charts

Create side-by-side pie & donut chart dengan:
- 5 categories
- Custom colors
- Percentage labels
- Explode one slice (pie only)

### Latihan 2: Complex Box Plot

Create box plot dengan:
- 4 groups
- Custom colors
- Mean markers
- Proper labels & title
- Grid on y-axis

### Latihan 3: Multi-panel Figure

Create 2x3 subplot grid dengan:
- Different plot types
- Shared axes where appropriate
- Consistent styling
- Main suptitle

### Latihan 4: Publication-Ready Plot

Create publication-quality plot dengan:
- Large fonts (title, labels)
- High resolution (300 DPI)
- Proper annotations
- Legend positioning
- Save in multiple formats

---

## 🔗 Referensi

- [Matplotlib Figure & Axes](https://matplotlib.org/stable/api/figure_api.html)
- [Matplotlib Subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)
- [Matplotlib Colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
