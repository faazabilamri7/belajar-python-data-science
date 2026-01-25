---
title: Projects & Advanced Topics
description: Real-world statistical analysis projects
sidebar:
  order: 8
---

## 🎯 Project 1: Student Performance Analysis

Complete statistical analysis dari student test scores.

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataset
np.random.seed(42)
n_students = 150

df = pd.DataFrame({
    'student_id': range(1, n_students + 1),
    'class': np.random.choice(['A', 'B', 'C'], n_students),
    'study_hours': np.random.uniform(0.5, 15, n_students),
    'previous_gpa': np.random.uniform(2, 4, n_students),
    'test_score': np.random.normal(75, 12, n_students)
})

# Add relationships
df['test_score'] += df['study_hours'] * 2 + df['previous_gpa'] * 5

print("=== STUDENT PERFORMANCE ANALYSIS ===\n")

# 1. DESCRIPTIVE STATISTICS
print("1. DESCRIPTIVE STATISTICS")
print(df.describe())

# 2. DISTRIBUTION ANALYSIS
print("\n2. DISTRIBUTION ANALYSIS")

# Check normality
_, p_norm = stats.shapiro(df['test_score'])
skewness = stats.skew(df['test_score'])
kurtosis_val = stats.kurtosis(df['test_score'])

print(f"  Normality test: p={p_norm:.4f} {'✓ Normal' if p_norm > 0.05 else '✗ Not normal'}")
print(f"  Skewness: {skewness:.3f}")
print(f"  Kurtosis: {kurtosis_val:.3f}")

# 3. CORRELATION ANALYSIS
print("\n3. CORRELATION ANALYSIS")

numeric_cols = ['study_hours', 'previous_gpa', 'test_score']
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)

# Correlation with test_score
print("\nCorrelation with test_score:")
for col in ['study_hours', 'previous_gpa']:
    r, p = stats.pearsonr(df[col], df['test_score'])
    sig = "✓" if p < 0.05 else "✗"
    print(f"  {sig} {col}: r={r:.3f} (p={p:.6f})")

# 4. GROUP COMPARISON (ANOVA)
print("\n4. GROUP COMPARISON (Classes)")

class_A = df[df['class'] == 'A']['test_score']
class_B = df[df['class'] == 'B']['test_score']
class_C = df[df['class'] == 'C']['test_score']

print(f"  Class A: mean={class_A.mean():.2f}, std={class_A.std():.2f}")
print(f"  Class B: mean={class_B.mean():.2f}, std={class_B.std():.2f}")
print(f"  Class C: mean={class_C.mean():.2f}, std={class_C.std():.2f}")

f_stat, p_anova = stats.f_oneway(class_A, class_B, class_C)
print(f"\nANOVA: F={f_stat:.4f}, p={p_anova:.4f}")
if p_anova < 0.05:
    print("  ✓ Classes have significantly different scores")

# 5. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution
axes[0, 0].hist(df['test_score'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['test_score'].mean(), color='r', linestyle='--', label=f'Mean: {df["test_score"].mean():.1f}')
axes[0, 0].set_title('Test Score Distribution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Scatter: Study Hours vs Score
axes[0, 1].scatter(df['study_hours'], df['test_score'], alpha=0.6)
axes[0, 1].set_xlabel('Study Hours')
axes[0, 1].set_ylabel('Test Score')
axes[0, 1].set_title(f'Study Hours vs Score (r={corr_matrix.loc["study_hours", "test_score"]:.3f})')
axes[0, 1].grid(alpha=0.3)

# Boxplot: Classes
df.boxplot(column='test_score', by='class', ax=axes[1, 0])
axes[1, 0].set_title('Test Scores by Class')
axes[1, 0].set_ylabel('Test Score')
axes[1, 0].get_figure().suptitle('')

# Q-Q Plot
stats.probplot(df['test_score'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)')

plt.tight_layout()
plt.show()

# 6. HYPOTHESIS TESTING
print("\n5. HYPOTHESIS TESTING")

# H₀: Mean score = 75
t_stat, p_val = stats.ttest_1samp(df['test_score'], 75)
print(f"\nH₀: Mean score = 75")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_val:.6f}")
if p_val < 0.05:
    print(f"  ✓ REJECT H₀: Mean score significantly different from 75")
else:
    print(f"  ✗ FAIL TO REJECT H₀")

print("\n✓ Analysis complete!")
```

---

## 🎯 Project 2: A/B Testing

Compare two versions of something (e.g., website versions).

```python
import numpy as np
import pandas as pd
from scipy import stats

# Simulate A/B test data
# Version A: original, Version B: new design
np.random.seed(42)

n = 1000

# Conversion rates
conversion_A = np.random.binomial(1, 0.10, n)  # 10% conversion
conversion_B = np.random.binomial(1, 0.12, n)  # 12% conversion

print("=== A/B TESTING ===\n")

print("Version A (Original):")
print(f"  Conversions: {conversion_A.sum()} / {n}")
print(f"  Conversion rate: {conversion_A.mean():.1%}")

print("\nVersion B (New Design):")
print(f"  Conversions: {conversion_B.sum()} / {n}")
print(f"  Conversion rate: {conversion_B.mean():.1%}")

# Chi-square test
contingency = np.array([
    [conversion_A.sum(), n - conversion_A.sum()],
    [conversion_B.sum(), n - conversion_B.sum()]
])

chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

print(f"\nChi-square Test:")
print(f"  χ² = {chi2:.4f}")
print(f"  p-value = {p_val:.4f}")

if p_val < 0.05:
    print(f"  ✓ Version B significantly different (p < 0.05)")
    
    # Calculate effect size
    if conversion_B.mean() > conversion_A.mean():
        improvement = (conversion_B.mean() - conversion_A.mean()) / conversion_A.mean() * 100
        print(f"  → {improvement:.1f}% improvement")
    else:
        print(f"  → Version B is worse")
else:
    print(f"  ✗ No significant difference (p >= 0.05)")
    print(f"  → Larger sample size needed or longer testing period")
```

---

## 🎯 Project 3: Multi-Group Analysis

Analyze performance across multiple groups.

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Dataset: Sales performance by region and product
np.random.seed(42)

data = {
    'region': np.repeat(['North', 'South', 'East', 'West'], 25),
    'product': np.tile(np.repeat(['A', 'B', 'C', 'D'], 6), 4) + 'A',  # Product A, B, C, D
    'sales': np.random.normal(100, 20, 100)
}

df = pd.DataFrame(data)
df['sales'] += pd.Categorical(df['region']).codes * 10  # Regional effect

print("=== MULTI-GROUP ANALYSIS ===\n")

# 1. Group means
print("1. GROUP MEANS\n")

region_means = df.groupby('region')['sales'].agg(['mean', 'std', 'count'])
print("By Region:")
print(region_means)

product_means = df.groupby('product')['sales'].agg(['mean', 'std', 'count'])
print("\nBy Product:")
print(product_means)

# 2. ANOVA: Regions
print("\n2. ANOVA - REGIONS")

groups = [group['sales'].values for name, group in df.groupby('region')]
f_stat, p_val = stats.f_oneway(*groups)

print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {p_val:.6f}")

if p_val < 0.05:
    print("  ✓ Regions have significantly different sales")
else:
    print("  ✗ No significant difference by region")

# 3. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot by region
df.boxplot(column='sales', by='region', ax=axes[0])
axes[0].set_title('Sales by Region')
axes[0].set_ylabel('Sales')

# Boxplot by product
df.boxplot(column='sales', by='product', ax=axes[1])
axes[1].set_title('Sales by Product')
axes[1].set_ylabel('Sales')

plt.tight_layout()
plt.show()
```

---

## 🎯 Challenge 1: Effect Size Calculation

Understand the difference between statistical significance and practical significance.

```python
def calculate_effect_size(group1, group2):
    """Calculate Cohen's d effect size"""
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return cohens_d

# Example
group_A = np.random.normal(100, 15, 100)
group_B = np.random.normal(102, 15, 100)

cohens_d = calculate_effect_size(group_A, group_B)

print(f"Cohen's d: {cohens_d:.3f}")

# Interpretation
if abs(cohens_d) < 0.2:
    print("  Negligible effect")
elif abs(cohens_d) < 0.5:
    print("  Small effect")
elif abs(cohens_d) < 0.8:
    print("  Medium effect")
else:
    print("  Large effect")

print("\nRemember:")
print("✓ Statistical significance ≠ Practical significance")
print("✓ Large p-value with small sample ≠ No effect")
print("✓ Small p-value with large sample ≠ Large effect")
```

---

## 🎯 Challenge 2: Power Analysis

Calculate statistical power - probability detecting true effect.

```python
from scipy import stats

# Parameters
effect_size = 0.5  # Cohen's d
alpha = 0.05  # Significance level
power = 0.80  # Desired power (1 - Beta)

# Calculate required sample size
# Using statsmodels
try:
    from statsmodels.stats.power import ttest_power, tt_solve_power
    
    n_needed = tt_solve_power(
        effect_size=effect_size,
        nobs=None,  # Solving for sample size
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )
    
    print(f"Sample size needed per group: {int(np.ceil(n_needed))}")
    print(f"  Effect size: {effect_size}")
    print(f"  Alpha: {alpha}")
    print(f"  Power: {power}")
    
except ImportError:
    print("Install statsmodels: pip install statsmodels")
```

---

## ✏️ Latihan Mandiri

### Latihan 1: Full Analysis

Pilih dataset, lakukan:
1. Descriptive statistics
2. Check normality
3. Correlation analysis
4. Hypothesis testing (sesuai pertanyaan)
5. Visualizations
6. Report findings

### Latihan 2: A/B Testing

Design and analyze A/B test untuk:
- Website design
- Email subject line
- Pricing strategy

Include:
- Sample size calculation
- Power analysis
- Results interpretation

### Latihan 3: Multi-Group Analysis

Compare 3+ groups, use ANOVA + post-hoc testing

### Latihan 4: Presentation

Present findings dengan:
- Clear hypothesis
- Methodology
- Results & interpretation
- Recommendations

---

## 📝 Ringkasan - Best Practices

### Statistical Analysis Checklist

- [ ] Define research question clearly
- [ ] State null & alternative hypotheses
- [ ] Choose appropriate test (check assumptions!)
- [ ] Check sample size adequacy
- [ ] Run analysis with clear interpretation
- [ ] Report effect sizes, not just p-values
- [ ] Visualize results
- [ ] Discuss limitations
- [ ] Make actionable recommendations

### Common Pitfalls to Avoid

❌ **Don't:**
- Use mean/t-test without checking normality
- Report p-value without effect size
- Do multiple tests without correction
- Ignore confounding variables
- Make causal claims from correlations
- Use inappropriate test for data type

✅ **Do:**
- Visualize data first
- Check assumptions before testing
- Report confidence intervals
- Interpret p-value correctly (probability of data under H₀)
- Distinguish correlation from causation
- Validate results with new data

---

## 🔗 Referensi

- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [StatsModels Documentation](https://www.statsmodels.org/)
- [Statistical Testing Guide](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
