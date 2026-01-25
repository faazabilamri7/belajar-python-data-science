---
title: Hypothesis Testing & Statistical Tests
description: P-value, T-test, ANOVA, Chi-Square
sidebar:
  order: 7
---

## 🧪 Apa itu Hypothesis Testing?

**Hypothesis Testing** adalah framework untuk membuat **data-driven decisions** menggunakan statistik. Kita test apakah ada evidence yang cukup untuk "reject" atau "fail to reject" null hypothesis.

### Konsep Dasar

**Null Hypothesis (H₀)**: Status quo, tidak ada effect
**Alternative Hypothesis (H₁)**: Ada effect/difference

```python
# Example: Testing apakah mean nilai siswa = 75?

# H₀: μ = 75 (mean nilai adalah 75)
# H₁: μ ≠ 75 (mean nilai BUKAN 75)

# Kita test H₀ menggunakan data
```

### P-value

**P-value** adalah probabilitas mengobservasi data seektrem ini (atau lebih ekstrem) **jika H₀ benar**.

- **P-value < 0.05**: Cukup extreme → Reject H₀ (ada evidence effect)
- **P-value ≥ 0.05**: Tidak extreme → Fail to reject H₀ (tidak cukup evidence)

```
P-value kecil = data unlikely under H₀ = reject H₀
P-value besar = data likely under H₀ = fail to reject H₀
```

---

## 📊 1. One-Sample T-Test

Test apakah **mean dari 1 sample = specific value**.

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

### Example: Student Test Scores

```python
from scipy import stats
import numpy as np

# Data: student test scores
scores = [78, 82, 90, 85, 88, 72, 95, 88, 79, 91]

# H₀: μ = 85 (mean score adalah 85)
# H₁: μ ≠ 85 (mean score bukan 85)

t_statistic, p_value = stats.ttest_1samp(scores, popmean=85)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Reject H₀: Mean score significantly different from 85")
else:
    print("✗ Fail to reject H₀: No evidence mean score ≠ 85")
```

### Interpretasi

```python
# Calculate sample statistics
sample_mean = np.mean(scores)
sample_std = np.std(scores, ddof=1)
n = len(scores)
se = sample_std / np.sqrt(n)  # Standard error

print(f"Sample mean: {sample_mean:.2f}")
print(f"Hypothesized mean: 85.00")
print(f"Standard error: {se:.2f}")
print(f"Difference: {sample_mean - 85:.2f}")

# Interpret p-value
print(f"\nP-value: {p_value:.4f}")
print(f"Interpretation: Probability observing mean = {sample_mean:.2f}")
print(f"  (or more extreme) if true mean = 85 is {p_value*100:.2f}%")
```

---

## 📊 2. Two-Sample T-Test

Test apakah **mean dari 2 different samples berbeda signifikan**.

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

### Example: Compare Two Classes

```python
# Test scores dari 2 kelas berbeda
class_A = [78, 82, 90, 85, 88, 92, 95, 88, 79, 91]
class_B = [72, 75, 70, 73, 68, 71, 74, 69, 72, 75]

# H₀: μₐ = μᵦ (mean scores same)
# H₁: μₐ ≠ μᵦ (mean scores different)

t_statistic, p_value = stats.ttest_ind(class_A, class_B)

print(f"Class A mean: {np.mean(class_A):.2f}")
print(f"Class B mean: {np.mean(class_B):.2f}")
print(f"\nT-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Classes have significantly different mean scores")
else:
    print("✗ No evidence classes have different mean scores")
```

### Assumptions Check

```python
# Check normality (assumption for t-test)
_, p_norm_A = stats.shapiro(class_A)
_, p_norm_B = stats.shapiro(class_B)

print(f"Normality test - Class A: p={p_norm_A:.4f}")
print(f"Normality test - Class B: p={p_norm_B:.4f}")

if p_norm_A < 0.05 or p_norm_B < 0.05:
    print("\n⚠ Warning: Data may not be normally distributed")
    print("Consider using Mann-Whitney U test (non-parametric)")
    
    # Mann-Whitney U test (non-parametric alternative)
    u_statistic, p_value_mw = stats.mannwhitneyu(class_A, class_B)
    print(f"\nMann-Whitney U: statistic={u_statistic:.4f}, p-value={p_value_mw:.4f}")
```

---

## 📊 3. ANOVA (Analysis of Variance)

Test apakah **mean dari 3+ groups berbeda signifikan**.

$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$

### Example: Compare 3 Classes

```python
class_A = [78, 82, 90, 85, 88, 92, 95, 88, 79, 91]
class_B = [72, 75, 70, 73, 68, 71, 74, 69, 72, 75]
class_C = [85, 88, 92, 90, 93, 87, 91, 89, 94, 86]

# H₀: μₐ = μᵦ = μ_c (all means same)
# H₁: At least one mean different

f_statistic, p_value = stats.f_oneway(class_A, class_B, class_C)

print(f"Class A mean: {np.mean(class_A):.2f}")
print(f"Class B mean: {np.mean(class_B):.2f}")
print(f"Class C mean: {np.mean(class_C):.2f}")
print(f"\nF-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("✓ At least one class has significantly different mean")
else:
    print("✗ No evidence classes have different means")
```

### Post-hoc Testing

Jika ANOVA significant, kita test **which pairs berbeda** (pairwise comparisons).

```python
from scipy.stats import ttest_ind

# Bonferroni correction untuk multiple tests
alpha = 0.05
n_comparisons = 3  # A vs B, A vs C, B vs C
alpha_corrected = alpha / n_comparisons

print("Pairwise Comparisons (with Bonferroni correction):")
print(f"Alpha corrected: {alpha_corrected:.4f}\n")

comparisons = [
    ('A', 'B', class_A, class_B),
    ('A', 'C', class_A, class_C),
    ('B', 'C', class_B, class_C)
]

for name1, name2, data1, data2 in comparisons:
    t_stat, p_val = ttest_ind(data1, data2)
    significant = "✓" if p_val < alpha_corrected else "✗"
    print(f"{significant} Class {name1} vs {name2}: p={p_val:.4f}")
```

---

## 📊 4. Chi-Square Test

Test apakah **categorical variables independent** (no association).

$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

### Example: Student Grade vs Gender

```python
# Contingency table: Grade by Gender
import pandas as pd

data = {
    'A': [20, 15],  # Males with A, Females with A
    'B': [25, 30],  # Males with B, Females with B
    'C': [15, 25],  # Males with C, Females with C
}

contingency_table = pd.DataFrame(data, index=['Male', 'Female'])
print("Contingency Table:")
print(contingency_table)

# Chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table.T)

print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("\n✓ Gender and Grade are associated")
else:
    print("\n✗ No evidence gender and grade are associated")

print("\nExpected Frequencies:")
print(expected)
```

---

## 🎯 Choosing the Right Test

### Decision Tree

```
What's your question?

1. Comparing 1 sample mean to a value?
   → One-sample t-test

2. Comparing 2 sample means?
   ├─ Data normal + equal variance?
   │  └─ t-test
   └─ Data NOT normal?
      └─ Mann-Whitney U test

3. Comparing 3+ group means?
   ├─ Data normal?
   │  └─ ANOVA
   └─ Data NOT normal?
      └─ Kruskal-Wallis test

4. Comparing categorical frequencies?
   └─ Chi-square test

5. Correlation between 2 variables?
   ├─ Linear + Normal?
   │  └─ Pearson r
   └─ Non-linear OR NOT Normal?
      └─ Spearman rho
```

---

## 📊 Example: Complete Analysis

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Create dataset
np.random.seed(42)
df = pd.DataFrame({
    'student_id': range(1, 101),
    'study_hours': np.random.uniform(1, 10, 100),
    'test_score': np.random.normal(75, 10, 100)
})

# Add relationship
df['test_score'] += df['study_hours'] * 3 + np.random.normal(0, 5, 100)

print("=== COMPLETE ANALYSIS ===\n")

# 1. Descriptive Statistics
print("1. DESCRIPTIVE STATISTICS")
print(df.describe())

# 2. Check Normality
print("\n2. NORMALITY CHECK (Shapiro-Wilk Test)")
_, p_norm_score = stats.shapiro(df['test_score'])
_, p_norm_hours = stats.shapiro(df['study_hours'])
print(f"  Test scores: p={p_norm_score:.4f} {'✓ Normal' if p_norm_score > 0.05 else '✗ Not normal'}")
print(f"  Study hours: p={p_norm_hours:.4f} {'✓ Normal' if p_norm_hours > 0.05 else '✗ Not normal'}")

# 3. Correlation Test
print("\n3. CORRELATION TEST (Pearson)")
r, p_corr = stats.pearsonr(df['study_hours'], df['test_score'])
print(f"  r = {r:.4f}, p-value = {p_corr:.6f}")
if p_corr < 0.05:
    print(f"  ✓ Study hours and test score are significantly correlated")

# 4. Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scatter plot
axes[0].scatter(df['study_hours'], df['test_score'], alpha=0.6)
axes[0].set_xlabel('Study Hours')
axes[0].set_ylabel('Test Score')
axes[0].set_title(f'Correlation: r={r:.3f}')
axes[0].grid(alpha=0.3)

# Distribution
axes[1].hist(df['test_score'], bins=20, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Test Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Test Scores')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## ⚠️ Common Mistakes

### 1. P-hacking (Multiple Testing)

```python
# ✗ WRONG: Test many hypotheses, report only significant ones
# ✓ CORRECT: Pre-specify hypotheses before analyzing

# Apply Bonferroni correction if testing multiple hypotheses
alpha = 0.05
n_tests = 5
alpha_corrected = alpha / n_tests
print(f"Use alpha = {alpha_corrected:.4f} instead of 0.05")
```

### 2. Ignoring Assumptions

```python
# ✗ WRONG: Use t-test without checking normality
# ✓ CORRECT: Check assumptions first

# If data not normal → use Mann-Whitney U or bootstrap
```

### 3. Small Sample Size

```python
# ✗ WRONG: n=3, report p-value
# ✓ CORRECT: Large n provides more reliable results

# Statistical power depends on sample size!
```

---

## 📝 Ringkasan

### Common Tests & When to Use

| Test | Purpose | Assumptions |
| ---- | ------- | ----------- |
| **One-sample t** | Mean = value | Normal or n > 30 |
| **Two-sample t** | Mean₁ = Mean₂ | Normal or n > 30 |
| **ANOVA** | 3+ means equal | Normal, equal variance |
| **Chi-square** | Categorical independence | n > 5 in cells |
| **Pearson r** | Correlation | Normal, linear |
| **Mann-Whitney U** | Non-parametric t-test | No assumptions |
| **Kruskal-Wallis** | Non-parametric ANOVA | No assumptions |

---

## ✏️ Latihan

### Latihan 1: One-Sample T-Test

Test apakah mean dataset = 100?

### Latihan 2: Two-Sample T-Test

Compare means dari 2 groups, interpret results

### Latihan 3: ANOVA

Compare 3+ groups, perform post-hoc testing

### Latihan 4: Chi-Square

Test association between categorical variables

---

## 🔗 Referensi

- [SciPy Stats Tests](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Statistical Testing Guide](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
