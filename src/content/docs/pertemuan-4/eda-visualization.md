---
title: EDA Visualization
description: Visualisasi data untuk exploratory analysis
sidebar:
  order: 3
---

## 📊 Visualisasi dalam EDA

Visualisasi adalah cara paling powerful untuk mengerti data. Dengan visualisasi yang tepat, kita bisa menemukan pattern, outliers, dan relationships yang tidak terlihat dari angka-angka raw.

### Setup Plotting

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = sns.load_dataset('titanic')
```

---

## 📈 1. Distribusi Data Univariat

Melihat distribusi satu variabel untuk memahami shape, central tendency, dan spread.

### Histogram

Histogram menunjukkan frekuensi data dalam bins (interval).

```python
# Single histogram
plt.figure(figsize=(10, 5))
plt.hist(df['age'].dropna(), bins=30, edgecolor='black', color='skyblue')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()

# Multiple histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df['age'].dropna(), bins=30, edgecolor='black')
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')

axes[1].hist(df['fare'].dropna(), bins=30, edgecolor='black')
axes[1].set_title('Fare Distribution')
axes[1].set_xlabel('Fare')

plt.tight_layout()
plt.show()

# Histogram untuk semua numeric columns
df.hist(figsize=(12, 10), bins=20)
plt.tight_layout()
plt.show()
```

### Density Plot (KDE)

Density plot adalah smoothed version dari histogram.

```python
# Density plot
df['age'].dropna().plot(kind='density', figsize=(10, 5))
plt.xlabel('Age')
plt.title('Age Density Distribution')
plt.show()

# Histogram dengan density overlay
plt.figure(figsize=(10, 5))
df['age'].dropna().hist(bins=30, density=True, alpha=0.7, edgecolor='black')
df['age'].dropna().plot(kind='density', color='red', linewidth=2)
plt.xlabel('Age')
plt.title('Age Distribution with Density')
plt.legend(['Density', 'Histogram'])
plt.show()

# Seaborn distplot
plt.figure(figsize=(10, 5))
sns.histplot(df['age'].dropna(), kde=True, bins=30)
plt.title('Age Distribution')
plt.show()
```

### Boxplot

Boxplot menunjukkan quartiles, median, dan outliers.

```python
# Single boxplot
plt.figure(figsize=(8, 5))
df['age'].plot(kind='box')
plt.title('Age Boxplot')
plt.show()

# Multiple boxplots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df['age'].plot(kind='box')
plt.title('Age Boxplot')

plt.subplot(1, 2, 2)
df['fare'].plot(kind='box')
plt.title('Fare Boxplot')

plt.tight_layout()
plt.show()

# Seaborn boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='sex', y='age')
plt.title('Age Distribution by Sex')
plt.show()
```

---

## 📊 2. Analisis Kategorikal

Untuk kolom kategorikal, kita perlu melihat distribution dari setiap kategori.

### Bar Chart

```python
# Value counts bar chart
plt.figure(figsize=(10, 5))
df['sex'].value_counts().plot(kind='bar', color=['steelblue', 'coral'])
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Distribution of Sex')
plt.xticks(rotation=0)
plt.show()

# Multiple bar charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df['sex'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Sex Distribution')

df['pclass'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Passenger Class Distribution')

plt.tight_layout()
plt.show()

# Seaborn countplot
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='sex', palette='Set2')
plt.title('Distribution of Sex')
plt.show()
```

### Pie Chart

```python
# Pie chart
plt.figure(figsize=(8, 6))
df['sex'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.ylabel('')  # Remove default ylabel
plt.title('Proportion of Sex')
plt.show()

# Multiple pie charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df['sex'].value_counts().plot(kind='pie', ax=axes[0], autopct='%1.1f%%')
axes[0].set_title('Sex Distribution')
axes[0].set_ylabel('')

df['pclass'].value_counts().sort_index().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_title('Passenger Class')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

---

## 🔗 3. Hubungan Bivariat

Melihat hubungan antara dua variabel untuk menemukan correlation dan dependencies.

### Scatter Plot

```python
# Simple scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['age'].dropna(), df['fare'].dropna(), alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare')
plt.show()

# Scatter plot dengan color coding
plt.figure(figsize=(10, 6))
colors = {'male': 'blue', 'female': 'red'}
for sex in df['sex'].unique():
    mask = df['sex'] == sex
    plt.scatter(df[mask]['age'], df[mask]['fare'], 
                label=sex, alpha=0.6, c=colors[sex])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()
plt.title('Age vs Fare by Sex')
plt.show()

# Seaborn scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='fare', hue='sex', alpha=0.6)
plt.title('Age vs Fare')
plt.show()
```

### Line Plot (untuk time series)

```python
# Time series plot
daily_survival = df.groupby('pclass')['survived'].mean()
plt.figure(figsize=(10, 6))
daily_survival.plot(marker='o', linewidth=2)
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class')
plt.grid(True, alpha=0.3)
plt.show()
```

### Violin Plot

Violin plot menggabungkan box plot dengan density plot.

```python
# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='sex', y='age')
plt.title('Age Distribution by Sex')
plt.show()

# Multiple violin plots
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='pclass', y='age', hue='sex')
plt.title('Age Distribution by Class and Sex')
plt.show()
```

---

## 🔥 4. Correlation Analysis

Melihat korelasi antar variabel numerik untuk menemukan relationships.

### Correlation Matrix

```python
# Calculate correlation
corr_matrix = df.select_dtypes(include=[np.number]).corr()
print(corr_matrix)

# Output:
#          survived    pclass       age      fare
# survived   1.000000 -0.338846  0.071949  0.257307
# pclass    -0.338846  1.000000 -0.369226 -0.549360
# age        0.071949 -0.369226  1.000000  0.096067
# fare       0.257307 -0.549360  0.096067  1.000000
```

### Correlation Heatmap

```python
# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Heatmap tanpa diagonal
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
            center=0, fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation Heatmap (Lower Triangle)')
plt.show()
```

### Pairplot

```python
# Pairplot - scatter plot untuk semua kombinasi
plt.figure(figsize=(12, 10))
# Hanya numeric columns
df_numeric = df[['age', 'fare', 'pclass', 'survived']].dropna()
sns.pairplot(df_numeric, diag_kind='hist', hue=None)
plt.show()

# Dengan color by category
sns.pairplot(df[['age', 'fare', 'sex', 'survived']].dropna(), 
             hue='sex', diag_kind='kde')
plt.show()
```

---

## 🔀 5. Segmented Analysis

Visualisasi yang membandingkan distribution across groups.

### Grouped Histograms

```python
# Histogram grouped by category
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, sex in enumerate(df['sex'].unique()):
    axes[i].hist(df[df['sex'] == sex]['age'].dropna(), bins=20, edgecolor='black')
    axes[i].set_title(f'Age Distribution - {sex.capitalize()}')
    axes[i].set_xlabel('Age')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Overlapping histograms
plt.figure(figsize=(10, 6))
for sex in df['sex'].unique():
    plt.hist(df[df['sex'] == sex]['age'].dropna(), bins=20, 
             alpha=0.5, label=sex)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.title('Age Distribution by Sex')
plt.show()
```

### Grouped Boxplots

```python
# Boxplot grouped
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='sex', y='age')
plt.title('Age by Sex')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='pclass', y='fare')
plt.title('Fare by Class')

plt.tight_layout()
plt.show()
```

---

## 📊 6. Missing Values Visualization

Visualisasi untuk melihat pattern dari missing values.

```python
# Heatmap of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Missing values per column
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Count': missing,
    'Percentage': missing_pct
}).sort_values('Percentage', ascending=False)

plt.figure(figsize=(10, 6))
missing_df[missing_df['Count'] > 0].plot(kind='bar', y='Percentage')
plt.ylabel('Percentage Missing (%)')
plt.title('Missing Values by Column')
plt.xticks(rotation=45)
plt.show()
```

---

## 🎨 7. Custom Visualization Template

Template untuk membuat comprehensive EDA visualization report.

```python
def plot_eda_report(df, numeric_cols=None, categorical_cols=None):
    """Generate comprehensive EDA report"""
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 1. Distribution of numeric columns
    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, 4*len(numeric_cols)))
    
    for i, col in enumerate(numeric_cols):
        # Histogram
        axes[i, 0].hist(df[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        axes[i, 0].set_title(f'{col} - Histogram')
        axes[i, 0].set_xlabel(col)
        
        # Boxplot
        axes[i, 1].boxplot(df[col].dropna())
        axes[i, 1].set_title(f'{col} - Boxplot')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Value counts for categorical columns
    fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(10, 3*len(categorical_cols)))
    
    for i, col in enumerate(categorical_cols):
        df[col].value_counts().plot(kind='bar', ax=axes[i], color='steelblue')
        axes[i].set_title(f'{col} - Distribution')
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.show()

# Gunakan
plot_eda_report(df)
```

---

## 📝 Ringkasan Halaman Ini

### Visualization Types

| Type | Use Case |
| ---- | --------- |
| Histogram | Distribusi variabel numerik |
| Boxplot | Quartiles, median, outliers |
| Scatter | Relasi 2 variabel numerik |
| Bar Chart | Frekuensi kategori |
| Pie Chart | Proporsi kategori |
| Heatmap | Korelasi atau missing values |
| Violin Plot | Distribusi per group |

---

## ✏️ Latihan

### Latihan 1: Histograms & Boxplots

```python
df = sns.load_dataset('iris')

# 1. Histogram untuk sepal_length
plt.hist(df['sepal_length'], bins=20, edgecolor='black')
plt.show()

# 2. Boxplot
df['sepal_length'].plot(kind='box')
plt.show()

# 3. Density plot
df['sepal_length'].plot(kind='density')
plt.show()
```

### Latihan 2: Categorical Analysis

```python
# 1. Bar chart
df['species'].value_counts().plot(kind='bar')
plt.show()

# 2. Pie chart
df['species'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()

# 3. Countplot
sns.countplot(data=df, x='species')
plt.show()
```

### Latihan 3: Relationships

```python
# 1. Scatter plot
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.show()

# 2. Correlation heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()

# 3. Boxplot by species
sns.boxplot(data=df, x='species', y='sepal_length')
plt.show()
```

---

## 🔗 Referensi

- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Pandas Plotting](https://pandas.pydata.org/docs/user_guide/visualization.html)
