---
title: Data Preparation
description: Preprocessing, feature engineering, dan data splitting
sidebar:
  order: 4
---

## 🔧 Data Preparation Pipeline

Data quality sangat mempengaruhi model performance. Berikut adalah pipeline standard untuk prepare data sebelum training:

```
Raw Data
   ↓
Handle Missing Values
   ↓
Encode Categorical
   ↓
Feature Scaling
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Ready untuk Training
```

---

## 1️⃣ Explore Data

Sebelum prepare, kita harus explore data untuk understand structure dan quality.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Basic info
print("Shape:", df.shape)                    # Baris dan kolom
print("\nInfo:")
print(df.info())                             # Tipe data, non-null count
print("\nDescriptive stats:")
print(df.describe())                         # Mean, std, min, max, quartiles
print("\nMissing values:")
print(df.isnull().sum())                     # Missing count per column
print("\nData types:")
print(df.dtypes)                             # Data type setiap kolom
print("\nTarget distribution:")
print(df['target'].value_counts())           # Class distribution
```

---

## 2️⃣ Handle Missing Values

### Identify Missing Values

```python
# Mana kolom yang punya missing?
missing = df.isnull().sum()
print(missing[missing > 0])

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
missing.plot(kind='barh')
plt.title('Missing Values per Column')
plt.show()

# Percentage missing
print(df.isnull().sum() / len(df) * 100)
```

### Handling Strategies

```python
# Strategy 1: Drop rows dengan missing
df_clean = df.dropna()  # Drop ANY row dengan missing
df_clean = df.dropna(subset=['important_col'])  # Drop jika col tertentu missing

# Strategy 2: Drop columns (jika kebanyakan missing)
df_clean = df.drop(columns=['col_with_95_missing'])

# Strategy 3: Fill dengan mean/median (untuk numeric)
df['numeric_col'].fillna(df['numeric_col'].mean(), inplace=True)
df['numeric_col'].fillna(df['numeric_col'].median(), inplace=True)

# Strategy 4: Fill dengan mode (untuk categorical)
df['category_col'].fillna(df['category_col'].mode()[0], inplace=True)

# Strategy 5: Forward fill / Backward fill (untuk time series)
df['column'].fillna(method='ffill', inplace=True)  # Forward fill
df['column'].fillna(method='bfill', inplace=True)  # Backward fill

# Strategy 6: Advanced (interpolation)
df['column'].interpolate(method='linear', inplace=True)

# Check hasil
print(df.isnull().sum())  # Semoga 0!
```

**Kapan pakai apa?**

| Strategy | When | Pros | Cons |
| -------- | ---- | ---- | ---- |
| Drop rows | <5% missing | Simple, clean | Lose data |
| Drop columns | >50% missing | Simple | Lose features |
| Mean/Median | Numeric, MCAR | Simple, preserve data | Reduce variance |
| Mode | Categorical | Simple | Lose info |
| Interpolation | Time series | Good for trends | Assume patterns |

---

## 3️⃣ Encode Categorical Variables

Model ML hanya paham angka, jadi categorical harus di-encode menjadi numeric.

### Strategy 1: One-Hot Encoding

Convert kategori menjadi binary columns.

```python
# Before:
# color: ['red', 'blue', 'green']

# After one-hot:
# color_red:   [1, 0, 0]
# color_blue:  [0, 1, 0]
# color_green: [0, 0, 1]

# Implementasi
df = pd.get_dummies(df, columns=['color'], drop_first=True)
# drop_first=True untuk avoid multicollinearity

# Result:
# color_blue:   [0, 1, 0]
# color_green:  [0, 0, 1]
# (color_red otomatis = 0,0)
```

**Kapan pakai:**
- ✅ Categorical dengan few unique values (<10)
- ✅ Tidak ada ordinal relationship
- ✅ Tree-based models, neural networks

### Strategy 2: Label Encoding

Assign numeric label ke setiap kategori.

```python
# Before: ['low', 'medium', 'high']
# After:  [0, 1, 2]

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['size'] = encoder.fit_transform(df['size'])

print(encoder.classes_)  # ['high', 'low', 'medium']
```

**Kapan pakai:**
- ✅ Categorical dengan ordinal relationship (low < medium < high)
- ✅ Tree-based models
- ❌ Jangan untuk linear models (akan assume 2 > 1)

### Strategy 3: Target Encoding

Encode dengan mean target value.

```python
# Sebelum: color=['red', 'blue', 'green']
# Setelah: color=[0.7, 0.3, 0.6]  (mean target value per color)

target_encoding = df.groupby('color')['target'].mean()
df['color_encoded'] = df['color'].map(target_encoding)
```

**Kapan pakai:**
- ✅ Categorical dengan many unique values
- ✅ Need encode specific category info
- ⚠️ Risk: Overfitting jika tidak careful

### Complete Example

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'color': ['red', 'blue', 'red', 'green'],
    'size': ['small', 'medium', 'large', 'small'],
    'target': [0, 1, 1, 0]
})

# Method 1: One-hot encoding
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)

# Method 2: Label encoding untuk size (ordinal)
size_order = ['small', 'medium', 'large']
df['size'] = df['size'].map({v: k for k, v in enumerate(size_order)})

print(df_encoded)
```

---

## 4️⃣ Feature Scaling

Scale numeric features ke range yang sama untuk algorithm yang distance-based (KNN, SVM, Neural Networks).

### Why Scale?

```
Feature 1: age (range 0-100)
Feature 2: income (range 0-10,000,000)

Jika tidak scale:
- Income dominates distance calculation
- Age jadi insignificant
- Model bias ke income

Dengan scale:
- Semua features punya weight sama
- Fair comparison
```

### Standard Scaling (Standardization)

Transform ke mean=0, std=1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit pada training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply same transformation ke test data
X_test_scaled = scaler.transform(X_test)

# Formula: (x - mean) / std
```

### Min-Max Scaling (Normalization)

Scale ke range [0, 1].

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Formula: (x - min) / (max - min)
```

### Comparison

```
Original:       [1, 100, 50000]
StandardScaler: [-1.5, 1.0, 1.3]      (mean=0, std=1)
MinMaxScaler:   [0, 1, 1]             (range 0-1)
```

**Kapan pakai apa?**

| Scaler | When | Range | Sensitive to Outliers |
| ------ | ---- | ----- | --------------------- |
| StandardScaler | Default choice | (-∞, +∞) | No |
| MinMaxScaler | Need bounded range | [0, 1] | Yes |
| RobustScaler | Have outliers | (-∞, +∞) | No |

### Important Rules

```python
# ✅ CORRECT: Fit pada train, apply ke test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # USE SAME scaler!

# ❌ WRONG: Fit again pada test (leakage!)
scaler_test = StandardScaler()
X_test_scaled = scaler_test.fit_transform(X_test)  # Different means!
```

---

## 5️⃣ Feature Engineering

Buat fitur baru dari fitur existing yang lebih informative.

### Techniques

#### 1. Polynomial Features

Tambah polynomial transformations.

```python
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[2], [3], [4]])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Result:
# [[1, 2, 4],      (1, x, x²)
#  [1, 3, 9],
#  [1, 4, 16]]

# Useful untuk linear model dengan non-linear relationship
```

#### 2. Binning / Discretization

Convert continuous ke categorical ranges.

```python
# Age: [5, 15, 25, 35, ..., 95]
# Menjadi: ['child', 'teenager', 'adult', 'senior']

df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 13, 18, 65, 100],
                         labels=['child', 'teen', 'adult', 'senior'])
```

#### 3. Feature Interaction

Buat fitur dari kombinasi fitur lain.

```python
# Original: [height, weight]
# New features:
df['height_weight_ratio'] = df['height'] / df['weight']
df['bmi'] = df['weight'] / (df['height'] ** 2)
df['height_weight_interaction'] = df['height'] * df['weight']
```

#### 4. Domain-Specific Features

Gunakan domain knowledge untuk buat fitur.

```python
# Untuk time series:
df['hour'] = pd.to_datetime(df['datetime']).dt.hour
df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Untuk sales:
df['revenue_per_customer'] = df['revenue'] / df['num_customers']
df['discount_rate'] = df['discount'] / df['price']
```

#### 5. Feature Selection

Keep hanya features yang informative, drop yang redundant.

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 5 features
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Which features selected?
selected_features = X.columns[selector.get_support()]
print(selected_features)
```

---

## 6️⃣ Train-Test Split

**Critical step:** Pisahkan data untuk training dan testing sebelum preprocessing!

```python
from sklearn.model_selection import train_test_split

# IMPORTANT: Split SEBELUM preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% test, 80% train (typical ratio)
    random_state=42,         # For reproducibility
    stratify=y               # Keep class distribution (untuk classification)
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Verify distribution
print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")
```

**Common Ratios:**

| Scenario | Train | Val | Test |
| -------- | ----- | --- | ---- |
| Small dataset | 60% | 20% | 20% |
| Medium dataset | 70% | - | 30% |
| Large dataset | 80% | - | 20% |
| With validation | 60% | 20% | 20% |

### Cross-Validation

Alternative ke simple train-test split untuk better estimation.

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross validation
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 🔄 Complete Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load & Explore
df = pd.read_csv('data.csv')

# 2. Split (EARLY!)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Identify column types
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# 4. Create preprocessing pipeline
from sklearn.preprocessing import OneHotEncoder
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 5. Fit preprocessor pada training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 6. Ready untuk training!
model = RandomForestClassifier()
model.fit(X_train_processed, y_train)
```

---

## ✏️ Latihan

### Latihan 1: Missing Values

1. Load dataset dengan missing values
2. Identify kolom dengan missing
3. Try 3 different strategies (drop, mean, median)
4. Compare hasil

### Latihan 2: Feature Encoding

1. Load dataset dengan categorical columns
2. Apply one-hot encoding
3. Apply label encoding
4. Compare dimensionality

### Latihan 3: Feature Engineering

Create 5 new features dari existing features:

```python
df['feature1'] = ...
df['feature2'] = ...
df['feature3'] = ...
df['feature4'] = ...
df['feature5'] = ...
```

### Latihan 4: Complete Pipeline

1. Load data
2. Handle missing values
3. Encode categorical
4. Scale numeric features
5. Split train-test
6. Ready untuk training!

---

## 🔗 Referensi

- [Scikit-Learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas Data Cleaning](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Feature Engineering Best Practices](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
