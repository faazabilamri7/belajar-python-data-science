---
title: Projects & Real Data Analysis
description: Real-world visualization projects
sidebar:
  order: 7
---

## 🎯 Project 1: EDA - Titanic Dataset

Complete exploratory data analysis dengan visualisasi.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
titanic = sns.load_dataset('titanic')

print("Dataset info:")
print(titanic.info())
print(titanic.head())

# ===== UNIVARIATE ANALYSIS =====
print("\n=== UNIVARIATE ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Age distribution
sns.histplot(data=titanic, x='age', kde=True, ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Age Distribution')

# 2. Passenger class
sns.countplot(data=titanic, x='pclass', ax=axes[0, 1])
axes[0, 1].set_title('Passenger Class Count')

# 3. Survival rate
survival_counts = titanic['survived'].value_counts()
axes[1, 0].pie(survival_counts, labels=['Died', 'Survived'], autopct='%1.1f%%')
axes[1, 0].set_title('Survival Rate')

# 4. Fare distribution
sns.kdeplot(data=titanic, x='fare', ax=axes[1, 1], fill=True)
axes[1, 1].set_title('Fare Distribution')

plt.tight_layout()
plt.show()

# ===== BIVARIATE ANALYSIS =====
print("\n=== BIVARIATE ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Survived vs Sex
sns.countplot(data=titanic, x='sex', hue='survived', ax=axes[0, 0], palette='Set1')
axes[0, 0].set_title('Survival by Sex')

# 2. Age vs Survived
sns.boxplot(data=titanic, x='survived', y='age', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Age Distribution by Survival')

# 3. Class vs Survived
sns.barplot(data=titanic, x='pclass', y='survived', ax=axes[1, 0])
axes[1, 0].set_title('Survival Rate by Class')
axes[1, 0].set_ylabel('Survival Probability')

# 4. Fare vs Age (with survival color)
sns.scatterplot(data=titanic, x='age', y='fare', hue='survived', 
                alpha=0.6, palette='Set1', ax=axes[1, 1])
axes[1, 1].set_title('Fare vs Age by Survival')

plt.tight_layout()
plt.show()

# ===== MULTIVARIATE ANALYSIS =====
print("\n=== MULTIVARIATE ===")

# Survival by Class & Sex
fig, ax = plt.subplots(figsize=(10, 6))
pivot = titanic.pivot_table(values='survived', index='pclass', columns='sex')
pivot.plot(kind='bar', ax=ax, rot=0)
ax.set_title('Survival Rate by Class & Sex')
ax.set_ylabel('Survival Rate')
ax.set_xlabel('Passenger Class')
ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
plt.show()

# Key findings
print("\n=== KEY FINDINGS ===")
print(f"Survival rate: {titanic['survived'].mean():.1%}")
print(f"Female survival: {titanic[titanic['sex']=='female']['survived'].mean():.1%}")
print(f"Male survival: {titanic[titanic['sex']=='male']['survived'].mean():.1%}")
print(f"1st class survival: {titanic[titanic['pclass']==1]['survived'].mean():.1%}")
print(f"3rd class survival: {titanic[titanic['pclass']==3]['survived'].mean():.1%}")
```

---

## 🎯 Project 2: Time Series Analysis

Visualize trends dan patterns dalam time series data.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load flights dataset (monthly airline passenger data)
flights = sns.load_dataset('flights')

print(flights.head())

# ===== TIME SERIES TRENDS =====
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 1. Overall trend
sns.lineplot(data=flights, x='year', y='passengers', ax=axes[0], linewidth=2)
axes[0].set_title('Airline Passengers Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 2. Trend by month (multiple lines)
sns.lineplot(data=flights, x='year', y='passengers', hue='month', 
             ax=axes[1], palette='husl')
axes[1].set_title('Passengers by Month', fontsize=14, fontweight='bold')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== SEASONAL PATTERNS =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pivot untuk heatmap
pivot_data = flights.pivot(index='month', columns='year', values='passengers')

# Heatmap
sns.heatmap(pivot_data, cmap='RdYlGn', ax=axes[0], annot=False, cbar_kws={'label': 'Passengers'})
axes[0].set_title('Passenger Heatmap - Month vs Year')

# Box plot untuk compare months
sns.boxplot(data=flights, x='month', y='passengers', ax=axes[1], palette='Set2')
axes[1].set_title('Passenger Distribution by Month')
axes[1].set_ylabel('Passengers')

plt.tight_layout()
plt.show()

# ===== INSIGHTS =====
print("\n=== KEY PATTERNS ===")
avg_by_month = flights.groupby('month')['passengers'].mean()
print(f"Peak month: {avg_by_month.idxmax()} ({avg_by_month.max():.0f} passengers)")
print(f"Low month: {avg_by_month.idxmin()} ({avg_by_month.min():.0f} passengers)")
print(f"Overall growth: {flights[flights['year']==1960]['passengers'].mean():.0f} -> "
      f"{flights[flights['year']==1969]['passengers'].mean():.0f}")
```

---

## 🎯 Project 3: Statistical Comparison

Compare distributions & test for differences.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Create sample data - test scores untuk 3 groups
np.random.seed(42)

group_a = np.random.normal(75, 10, 100)
group_b = np.random.normal(78, 12, 100)
group_c = np.random.normal(72, 9, 100)

df = pd.DataFrame({
    'Score': np.concatenate([group_a, group_b, group_c]),
    'Group': ['A']*100 + ['B']*100 + ['C']*100
})

# ===== DISTRIBUTION COMPARISON =====
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histograms
for group in ['A', 'B', 'C']:
    axes[0, 0].hist(df[df['Group']==group]['Score'], 
                    alpha=0.5, label=f'Group {group}', bins=20)
axes[0, 0].set_title('Distribution Histograms')
axes[0, 0].set_xlabel('Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# KDE plots
sns.kdeplot(data=df, x='Score', hue='Group', ax=axes[0, 1], fill=True)
axes[0, 1].set_title('KDE - Distribution Comparison')

# Box plots
sns.boxplot(data=df, x='Group', y='Score', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Box Plot - Quartiles Comparison')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Violin plots
sns.violinplot(data=df, x='Group', y='Score', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Violin Plot - Distribution Shape')

plt.tight_layout()
plt.show()

# ===== STATISTICAL TESTING =====
print("\n=== STATISTICS ===")
for group in ['A', 'B', 'C']:
    scores = df[df['Group']==group]['Score']
    print(f"Group {group}: Mean={scores.mean():.1f}, Std={scores.std():.1f}")

# ANOVA test
f_stat, p_value = stats.f_oneway(group_a, group_b, group_c)
print(f"\nANOVA: F={f_stat:.4f}, p-value={p_value:.4f}")
if p_value < 0.05:
    print("✓ Significant difference between groups")
else:
    print("✗ No significant difference")
```

---

## 🎯 Project 4: Correlation & Multivariate Analysis

Analyze relationships dalam multivariate dataset.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
iris = sns.load_dataset('iris')

print(iris.head())

# ===== CORRELATION ANALYSIS =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correlation matrix heatmap
corr = iris.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=axes[0])
axes[0].set_title('Correlation Matrix')

# Clustered heatmap
from scipy.cluster.hierarchy import dendrogram, linkage
g = sns.clustermap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
g.fig.suptitle('Clustered Correlation', fontsize=12)

# ===== PAIRPLOT =====
g = sns.pairplot(iris, hue='species', height=2.5, plot_kws={'alpha': 0.6})
g.fig.suptitle('Iris Dataset - Pairplot', y=1.00)
plt.show()

# ===== FACETGRID ANALYSIS =====
# Relationship by species
g = sns.FacetGrid(iris, col='species', height=4)
g.map(sns.scatterplot, 'sepal_length', 'sepal_width', alpha=0.6)
g.set_titles('Species: {col_name}')
plt.show()

# ===== INSIGHTS =====
print("\n=== STRONGEST CORRELATIONS ===")
# Get correlations flattened
corr_flat = corr.unstack()
corr_flat = corr_flat[corr_flat != 1.0]  # Exclude diagonal
top_corr = corr_flat.nlargest(5)
for (var1, var2), corr_val in top_corr.items():
    print(f"{var1} ↔ {var2}: {corr_val:.3f}")
```

---

## ✏️ Challenge Exercises

### Challenge 1: Sales Data Dashboard

Create 4-panel dashboard dengan:
- Sales trend over time (line plot)
- Sales by region (bar plot)
- Sales vs marketing spend (scatter)
- Sales distribution (histogram)

All dalam satu figure dengan consistent styling.

### Challenge 2: Customer Segmentation

Analyze customer dataset dengan:
- Pairplot untuk explore relationships
- Heatmap untuk correlations
- FacetGrid untuk segment comparison
- Key insights documented

### Challenge 3: Publication Figure

Create publication-quality plot dengan:
- High resolution (300 DPI)
- Professional styling
- Clear titles & labels
- Proper annotations
- Save in multiple formats

### Challenge 4: Interactive Dashboard

Create interactive visualization workflow:
- Load data
- Create multiple views
- Add interactivity (hover, selection)
- Use Plotly atau Bokeh

---

## 📝 Best Practices Checklist

### Before Creating Plot

- [ ] Define specific question to answer
- [ ] Explore data first (know your data)
- [ ] Identify variable types (numeric/categorical)
- [ ] Know audience (technical vs non-technical)

### During Visualization

- [ ] Clear, descriptive title
- [ ] Labeled axes dengan units
- [ ] Legend jika multiple series
- [ ] Appropriate scale (linear vs log)
- [ ] Color accessible (colorblind friendly)

### After Visualization

- [ ] Verify accuracy
- [ ] Check for misleading elements
- [ ] Add context/annotations jika needed
- [ ] Save high resolution untuk presentation
- [ ] Test readability di berbagai ukuran

---

## 🔗 Referensi & Resources

### Datasets untuk Practice

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning](https://archive.ics.uci.edu/)
- [Seaborn Built-in Datasets](https://seaborn.pydata.org/generated/seaborn.get_dataset_names.html)

### Visualization Resources

- [Seaborn Gallery](https://seaborn.pydata.org/examples.html)
- [Matplotlib Examples](https://matplotlib.org/gallery/index.html)
- [Edward Tufte Books](https://www.edwardtufte.com/)
- [Data Visualization Society](https://www.datavisualizationsociety.org/)

### Tools & Libraries

- **Matplotlib** - Low-level, full control
- **Seaborn** - Statistical plots, beautiful defaults
- **Plotly** - Interactive visualization
- **Bokeh** - Interactive, streaming data
- **Altair** - Grammar of graphics, declarative
