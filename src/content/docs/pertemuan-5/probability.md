---
title: Probabilitas & Distribusi Probabilitas
description: Konsep probabilitas dasar dan distribusi probabilitas
sidebar:
  order: 5
---

## 🎲 Apa itu Probabilitas?

**Probabilitas** adalah ukuran kemungkinan suatu event terjadi. Nilai probabilitas: **0 sampai 1**
- **P = 0**: Tidak mungkin terjadi
- **P = 0.5**: Sama mungkin terjadi atau tidak
- **P = 1**: Pasti terjadi

$$P(A) = \frac{\text{jumlah outcome yang favorable}}{\text{jumlah total outcome}}$$

### Contoh Sederhana

```python
# Dadu (6 sisi): P(genap)?
favorable_outcomes = [2, 4, 6]  # 3 outcomes
total_outcomes = [1, 2, 3, 4, 5, 6]  # 6 outcomes

p_genap = len(favorable_outcomes) / len(total_outcomes)
print(f"P(genap) = {p_genap:.2f}")  # 0.50

# Coin flip: P(heads)?
p_heads = 1 / 2
print(f"P(heads) = {p_heads:.2f}")  # 0.50
```

---

## 🔗 Aturan Probabilitas

### 1. Complement Rule

Probabilitas suatu event terjadi + tidak terjadi = 1

$$P(A) + P(\neg A) = 1$$
$$P(\neg A) = 1 - P(A)$$

```python
# Contoh: Probability dadu bukan 6?
p_six = 1/6
p_not_six = 1 - p_six
print(f"P(6): {p_six:.3f}")  # 0.167
print(f"P(not 6): {p_not_six:.3f}")  # 0.833

# Verify
print(f"Sum: {p_six + p_not_six}")  # 1.0
```

### 2. Addition Rule

Probability A atau B = P(A) + P(B) - P(A dan B)

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

```python
# Contoh: P(genap ATAU > 3) pada dadu?
# Genap: {2, 4, 6}
# > 3: {4, 5, 6}
# Genap DAN > 3: {4, 6}

p_genap = 3/6
p_lebih_3 = 3/6
p_genap_dan_lebih_3 = 2/6

p_genap_atau_lebih_3 = p_genap + p_lebih_3 - p_genap_dan_lebih_3
print(f"P(genap atau > 3) = {p_genap_atau_lebih_3:.3f}")  # 0.667

# Verify: {2, 4, 5, 6} = 4 dari 6
print(f"Verify: 4/6 = {4/6:.3f}")
```

### 3. Multiplication Rule (Independent)

Probability A dan B (independent) = P(A) × P(B)

$$P(A \cap B) = P(A) \times P(B)$$

```python
# Contoh: P(coin heads AND dadu genap)?
p_heads = 1/2
p_genap = 3/6

p_heads_dan_genap = p_heads * p_genap
print(f"P(heads dan genap) = {p_heads_dan_genap:.3f}")  # 0.250

# Verify: 1/2 × 1/2 = 1/4
print(f"Verify: 1/4 = {1/4:.3f}")
```

### 4. Conditional Probability

P(A | B) = Probability A terjadi GIVEN B sudah terjadi

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

```python
# Contoh: P(dadu > 3 | dadu genap)?
# Outcomes genap: {2, 4, 6}
# Outcomes genap DAN > 3: {4, 6}

p_lebih_3_given_genap = 2 / 3  # 2 favorable dari 3 genap outcomes
print(f"P(> 3 | genap) = {p_lebih_3_given_genap:.3f}")  # 0.667

# Formal calculation
p_lebih_3_dan_genap = 2/6
p_genap = 3/6
p_conditional = p_lebih_3_dan_genap / p_genap
print(f"P(> 3 | genap) = {p_conditional:.3f}")  # 0.667 ✓
```

---

## 📊 Distribusi Probabilitas

**Distribusi Probabilitas** mendeskripsikan bagaimana probabilitas terdistribusi di antara semua kemungkinan outcomes.

Ada 2 tipe utama:
1. **Discrete Probability Distribution** - Finite number of outcomes (dadu, coin)
2. **Continuous Probability Distribution** - Infinite outcomes (normal, exponential)

---

## 🎲 1. Binomial Distribution

Distribusi binomial untuk **n trials** dengan **2 possible outcomes** (success/failure) dan **probability p** untuk setiap trial.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

### Contoh: Coin Flips

```python
from scipy import stats

# Scenario: 5 coin flips, P(3 heads)?
n = 5  # jumlah flips
k = 3  # jumlah heads yang diinginkan
p = 0.5  # probability heads

prob = stats.binom.pmf(k, n, p)
print(f"P(3 heads dari 5 flips) = {prob:.4f}")  # 0.3125

# Probability untuk semua outcomes
print("\nSemua kemungkinan:")
for x in range(n + 1):
    prob_x = stats.binom.pmf(x, n, p)
    print(f"  P({x} heads) = {prob_x:.4f}")
```

### Plotting Binomial Distribution

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Different probabilities
n = 10
for p, ax in zip([0.3, 0.7], axes):
    x = np.arange(0, n + 1)
    pmf = stats.binom.pmf(x, n, p)
    
    ax.bar(x, pmf, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Successes')
    ax.set_ylabel('Probability')
    ax.set_title(f'Binomial Distribution (n={n}, p={p})')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 📈 2. Normal Distribution (Gaussian Distribution)

Normal distribution untuk **continuous data**. Sudah dipelajari di bagian Distributions.

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Normal distribution: μ=100, σ=15
mu = 100
sigma = 15

# Generate random samples
samples = np.random.normal(mu, sigma, 10000)

# Calculate probability P(85 < X < 115)
prob = stats.norm.cdf(115, mu, sigma) - stats.norm.cdf(85, mu, sigma)
print(f"P(85 < X < 115) = {prob:.4f}")  # ≈ 0.6827 (68%)

# Plot
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = stats.norm.pdf(x, mu, sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, where=(x >= 85) & (x <= 115), alpha=0.3, color='green', label='P(85 < X < 115)')
plt.axvline(mu, color='r', linestyle='--', alpha=0.5, label=f'μ={mu}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Normal Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## 🎯 3. Poisson Distribution

Poisson distribution untuk **counting** - jumlah events dalam fixed interval.

$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}$$

Digunakan ketika:
- Counting events dalam time/space interval
- Events rare tapi independent

### Contoh: Customer Arrivals

```python
# Scenario: Average 3 customers per hour
# Probability exactly 5 customers dalam 1 hour?

lambda_param = 3  # average events per interval
k = 5  # desired number of events

prob = stats.poisson.pmf(k, lambda_param)
print(f"P(5 customers | avg 3 per hour) = {prob:.4f}")

# Probability distribution
print("\nProbability untuk different customer counts:")
for x in range(10):
    prob_x = stats.poisson.pmf(x, lambda_param)
    print(f"  P({x} customers) = {prob_x:.4f}")
```

### Plot Poisson Distribution

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

lambdas = [1, 3, 7]
for ax, lambda_param in zip(axes, lambdas):
    x = np.arange(0, 15)
    pmf = stats.poisson.pmf(x, lambda_param)
    
    ax.bar(x, pmf, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Probability')
    ax.set_title(f'Poisson Distribution (λ={lambda_param})')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 📊 Comparing Distributions

```python
import pandas as pd

# Create comparison table
distributions = {
    'Distribution': ['Binomial', 'Normal', 'Poisson', 'Uniform'],
    'Type': ['Discrete', 'Continuous', 'Discrete', 'Continuous'],
    'Parameters': ['n, p', 'μ, σ', 'λ', 'a, b'],
    'Use Case': [
        'n trials, 2 outcomes',
        'Continuous measurements',
        'Counting events',
        'Random uniform outcome'
    ],
    'Example': [
        'Coin flips',
        'Height, test scores',
        'Customer arrivals',
        'Random numbers'
    ]
}

df_compare = pd.DataFrame(distributions)
print(df_compare.to_string(index=False))
```

---

## 🧮 Expected Value & Variance

### Expected Value (Mean)

Expected value adalah **average outcome** jika experiment diulang banyak kali.

$$E[X] = \sum x \cdot P(x)$$

```python
# Contoh: Fair dadu
# E[X] = 1×(1/6) + 2×(1/6) + ... + 6×(1/6)

outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6

expected_value = sum(x * p for x, p in zip(outcomes, probabilities))
print(f"Expected value (dadu): {expected_value:.2f}")  # 3.50

# Verify with formula
print(f"Formula (1+2+3+4+5+6)/6 = {sum(outcomes)/len(outcomes):.2f}")
```

### Variance

Variance dari probability distribution:

$$Var(X) = E[X^2] - (E[X])^2$$

```python
# Variance dadu
e_x = 3.5
e_x2 = sum(x**2 * p for x, p in zip(outcomes, probabilities))
variance = e_x2 - e_x**2

print(f"E[X²]: {e_x2:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Std Dev: {np.sqrt(variance):.2f}")
```

---

## 📝 Ringkasan

### Aturan Probabilitas

| Rule | Formula |
| ---- | ------- |
| **Complement** | P(A) + P(¬A) = 1 |
| **Addition** | P(A ∪ B) = P(A) + P(B) - P(A ∩ B) |
| **Multiplication** | P(A ∩ B) = P(A) × P(B) [independent] |
| **Conditional** | P(A\|B) = P(A ∩ B) / P(B) |

### Distribusi Umum

| Distribution | Use | Parameters |
| ------------ | --- | ---------- |
| **Binomial** | n trials, 2 outcomes | n, p |
| **Normal** | Continuous data | μ, σ |
| **Poisson** | Counting events | λ |
| **Uniform** | Random uniform | a, b |

---

## ✏️ Latihan

### Latihan 1: Probability Calculations

1. P(dadu ≤ 3)?
2. P(dadu > 2 AND coin heads)?
3. P(dadu > 4 | dadu genap)?

### Latihan 2: Binomial Distribution

Dari 10 pertanyaan multiple choice (4 pilihan), probabilitas:
1. Exactly 3 benar (random guessing)?
2. At least 5 benar?
3. More than 7 benar?

### Latihan 3: Real-world Poisson

Jika rata-rata 2 error per 1000 lines of code:
1. P(exactly 3 errors)?
2. P(0 errors)?
3. P(more than 4 errors)?

---

## 🔗 Referensi

- [SciPy Stats Distributions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Probability Concepts](https://en.wikipedia.org/wiki/Probability_distribution)
