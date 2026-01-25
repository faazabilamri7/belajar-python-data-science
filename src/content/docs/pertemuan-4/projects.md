---
title: Projects & Advanced Cases
description: Real-world EDA & data cleaning projects
sidebar:
  order: 8
---

## 🎯 Project 1: Customer Sales Analysis

Complete EDA dan cleaning untuk sales dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'customer_id': np.random.choice(range(1, 21), 100),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food'], 100),
    'amount': np.random.normal(100, 50, 100),
    'payment_method': np.random.choice(['Cash', 'Card', 'Online'], 100)
})

# Add some missing & outliers
df.loc[np.random.choice(100, 5, replace=False), 'amount'] = np.nan
df.loc[5, 'amount'] = 500  # Outlier

print("=== ORIGINAL DATA ===")
print(f"Shape: {df.shape}")
print(f"Missing: {df.isnull().sum().sum()}")

# 2. EXPLORATORY DATA ANALYSIS
print("\n=== EDA ===")

# Basic stats
print(df.describe())

# Missing analysis
print(f"\nMissing values:\n{df.isnull().sum()}")

# Distribution
print(f"\nCategory distribution:\n{df['category'].value_counts()}")

# 3. DATA CLEANING
print("\n=== DATA CLEANING ===")

# Handle missing
df['amount'].fillna(df['amount'].median(), inplace=True)

# Handle outliers
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
df['amount'] = df['amount'].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

# 4. FEATURE ENGINEERING
print("\n=== FEATURE ENGINEERING ===")

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.day_name()
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# 5. ANALYSIS
print("\n=== ANALYSIS ===")

# Top categories
print(f"Top categories:\n{df.groupby('category')['amount'].sum().sort_values(ascending=False)}")

# Average by payment method
print(f"\nAverage amount by payment:\n{df.groupby('payment_method')['amount'].mean()}")

# Trend by month
print(f"\nTotal sales by month:\n{df.groupby('month')['amount'].sum()}")

# 6. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Amount distribution
axes[0,0].hist(df['amount'], bins=20, edgecolor='black')
axes[0,0].set_title('Amount Distribution')

# By category
df.groupby('category')['amount'].sum().plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Sales by Category')

# By payment method
df.groupby('payment_method')['amount'].mean().plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Average Amount by Payment')

# Trend
df.groupby('month')['amount'].sum().plot(ax=axes[1,1], marker='o')
axes[1,1].set_title('Monthly Trend')

plt.tight_layout()
plt.show()

print("\n✓ Analysis complete!")
```

---

## 🎯 Project 2: Employee Data Cleaning

Complex dataset dengan berbagai data quality issues.

```python
import pandas as pd
import numpy as np

# Create dataset dengan issues
np.random.seed(42)
data = {
    'employee_id': range(1, 101),
    'name': ['Employee_' + str(i).upper() for i in range(1, 101)],
    'hire_date': pd.date_range('2020-01-01', periods=100),
    'salary': np.random.randint(50000, 150000, 100),
    'department': np.random.choice(['IT', 'HR', 'Finance', 'Sales'], 100),
    'gender': np.random.choice(['M', 'F', 'Other'], 100),
    'performance_score': np.random.uniform(1, 5, 100)
}

df = pd.DataFrame(data)

# Add data quality issues
# 1. Missing values
df.loc[np.random.choice(100, 10, replace=False), 'salary'] = np.nan
df.loc[np.random.choice(100, 5, replace=False), 'performance_score'] = np.nan

# 2. Duplicates
df = pd.concat([df, df.iloc[:5]], ignore_index=True)

# 3. Outliers
df.loc[5, 'salary'] = 500000
df.loc[10, 'performance_score'] = 10

# 4. Inconsistencies
df.loc[0, 'name'] = '  JOHN DOE  '  # Extra spaces
df.loc[1, 'gender'] = 'M '  # Trailing space

print("=== BEFORE CLEANING ===")
print(f"Shape: {df.shape}")
print(f"Missing:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# ===== CLEANING PIPELINE =====

# 1. Handle duplicates
df = df.drop_duplicates()
print(f"\n✓ After drop duplicates: {len(df)} rows")

# 2. Handle missing values
df['salary'].fillna(df['salary'].median(), inplace=True)
df['performance_score'].fillna(df.groupby('department')['performance_score'].transform('mean'), inplace=True)

# 3. Handle outliers
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
df['salary'] = df['salary'].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

df['performance_score'] = df['performance_score'].clip(1, 5)

# 4. Clean text
df['name'] = df['name'].str.strip().str.title()
df['gender'] = df['gender'].str.strip().str.upper()

# 5. Fix data types
df['hire_date'] = pd.to_datetime(df['hire_date'])

# 6. Feature engineering
df['years_employed'] = (pd.Timestamp.now() - df['hire_date']).dt.days / 365.25
df['salary_level'] = pd.cut(df['salary'], 
                             bins=[0, 75000, 100000, 150000],
                             labels=['Junior', 'Mid', 'Senior'])

print("\n=== AFTER CLEANING ===")
print(f"Shape: {df.shape}")
print(f"Missing: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Dtypes:\n{df.dtypes}")

# ===== ANALYSIS =====
print("\n=== ANALYSIS ===")

print(f"Average salary by department:\n{df.groupby('department')['salary'].mean().round(0)}")
print(f"\nAverage performance by salary level:\n{df.groupby('salary_level')['performance_score'].mean().round(2)}")
print(f"\nCount by gender:\n{df['gender'].value_counts()}")
```

---

## 🎯 Project 3: Time Series Data Quality

Handling missing values dan outliers dalam time series.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create time series with issues
dates = pd.date_range('2024-01-01', periods=100, freq='D')
np.random.seed(42)

values = np.random.normal(100, 10, 100)

df = pd.DataFrame({
    'date': dates,
    'value': values
})

# Add issues
# 1. Missing values (scattered)
df.loc[np.random.choice(100, 10, replace=False), 'value'] = np.nan

# 2. Outliers
df.loc[20, 'value'] = 200
df.loc[50, 'value'] = 50

print("=== BEFORE ===")
print(f"Missing: {df['value'].isnull().sum()}")
print(f"Mean: {df['value'].mean():.2f}")
print(f"Std: {df['value'].std():.2f}")

# ===== CLEANING FOR TIME SERIES =====

# 1. Forward fill for missing (preserve trend)
df['value_ffill'] = df['value'].fillna(method='ffill').fillna(method='bfill')

# 2. Interpolation (linear)
df['value_interp'] = df['value'].interpolate(method='linear')

# 3. Outlier capping
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['value_capped'] = df['value'].clip(lower, upper)

# ===== VISUALIZATION =====
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Original with missing/outliers
axes[0].plot(df['date'], df['value'], marker='o', alpha=0.7, label='Original')
axes[0].set_title('Original Data (with missing & outliers)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Forward fill
axes[1].plot(df['date'], df['value_ffill'], marker='o', alpha=0.7, label='Forward Fill')
axes[1].set_title('Forward Fill Method')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Interpolation
axes[2].plot(df['date'], df['value_interp'], marker='o', alpha=0.7, label='Interpolation')
axes[2].set_title('Linear Interpolation')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== COMPARISON =====
print("\n=== AFTER CLEANING ===")
print(f"Forward fill mean: {df['value_ffill'].mean():.2f}")
print(f"Interpolation mean: {df['value_interp'].mean():.2f}")
print(f"Capped mean: {df['value_capped'].mean():.2f}")
```

---

## 🎯 Challenge 1: Data Validation

Validate data berdasarkan business rules.

```python
def validate_data(df):
    """Validate data against business rules"""
    
    issues = []
    
    # Rule 1: Age harus antara 18-65
    if (df['age'] < 18).any() or (df['age'] > 65).any():
        invalid_count = ((df['age'] < 18) | (df['age'] > 65)).sum()
        issues.append(f"⚠ {invalid_count} rows dengan age invalid")
    
    # Rule 2: Salary harus positif
    if (df['salary'] <= 0).any():
        invalid_count = (df['salary'] <= 0).sum()
        issues.append(f"⚠ {invalid_count} rows dengan salary negatif")
    
    # Rule 3: Email harus valid format
    if 'email' in df.columns:
        valid_emails = df['email'].str.contains(r'\w+@\w+\.\w+', regex=True, na=False).sum()
        if valid_emails < len(df):
            issues.append(f"⚠ {len(df) - valid_emails} rows dengan email format invalid")
    
    # Rule 4: Hire date harus sebelum hari ini
    if 'hire_date' in df.columns:
        df['hire_date'] = pd.to_datetime(df['hire_date'])
        if (df['hire_date'] > pd.Timestamp.now()).any():
            invalid_count = (df['hire_date'] > pd.Timestamp.now()).sum()
            issues.append(f"⚠ {invalid_count} rows dengan hire_date di masa depan")
    
    return issues

# Test
if issues:
    print("Data Validation Issues:")
    for issue in issues:
        print(issue)
else:
    print("✓ All validation passed!")
```

---

## 🎯 Challenge 2: Outlier Investigation

Investigate outliers untuk understand root cause.

```python
def investigate_outliers(df, column):
    """Investigate outliers in detail"""
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    
    print(f"=== OUTLIER INVESTIGATION: {column} ===")
    print(f"Normal range: {lower:.2f} - {upper:.2f}")
    print(f"Outliers found: {len(outliers)}\n")
    
    if len(outliers) > 0:
        print("Outlier Details:")
        print(outliers.to_string())
        
        print(f"\nOutlier Statistics:")
        print(f"  Min: {outliers[column].min():.2f}")
        print(f"  Max: {outliers[column].max():.2f}")
        print(f"  Mean: {outliers[column].mean():.2f}")
        
        # Check if correlated dengan other columns
        print(f"\nCorrelation with other columns:")
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != column:
                corr = outliers[column].corr(outliers[col])
                print(f"  {col}: {corr:.3f}")

# Test
investigate_outliers(df, 'salary')
```

---

## 📝 Ringkasan Projects

### Project Outcomes

| Project | Key Learnings |
| ------- | ------------- |
| Sales Analysis | Date features, group analysis, visualization |
| Employee Cleaning | Complex pipeline, multiple issues, feature engineering |
| Time Series | Forward fill, interpolation, trend preservation |

---

## ✏️ Latihan Mandiri

### Latihan 1: Your First EDA

1. Download dataset dari Kaggle
2. Load dan inspect
3. Generate EDA report
4. Identify top 5 issues
5. Create cleaning plan

### Latihan 2: Custom Pipeline

1. Create dataset dengan issues
2. Build cleaning pipeline
3. Generate before/after report
4. Validate results

### Latihan 3: Advanced Analysis

1. Find meaningful patterns
2. Create derived features
3. Segment by categories
4. Make business recommendations

---

## 🔗 Referensi

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Real-world EDA Examples](https://www.kaggle.com/kernels?sortBy=voteCount&group=everyone&pageSize=20&datasetId=&submittedAfter=2020-01-01&language=python)
