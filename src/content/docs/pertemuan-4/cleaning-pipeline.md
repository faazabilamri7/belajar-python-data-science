---
title: Data Cleaning Pipeline
description: Template dan best practices untuk data cleaning
sidebar:
  order: 7
---

## 📋 Complete EDA & Cleaning Checklist

Gunakan checklist ini untuk memastikan data sudah siap untuk analisis.

```python
def eda_and_cleaning_checklist(df):
    """Complete EDA & cleaning checklist"""
    
    checklist = {
        "Load & Inspect": False,
        "Handle Missing": False,
        "Remove Duplicates": False,
        "Fix Data Types": False,
        "Handle Outliers": False,
        "Clean Text": False,
        "Feature Engineering": False,
        "Normalize/Scale": False,
        "Validate Final": False
    }
    
    print("=" * 60)
    print("EDA & DATA CLEANING CHECKLIST")
    print("=" * 60)
    
    # 1. LOAD & INSPECT
    print("\n✓ STEP 1: Load & Inspect")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"  - Dtypes:\n{df.dtypes}")
    print(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    checklist["Load & Inspect"] = True
    
    # 2. HANDLE MISSING
    print("\n✓ STEP 2: Handle Missing Values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  - Missing values found:")
        print(missing[missing > 0])
        # Handle each missing column
        # ...
    else:
        print("  - No missing values!")
    checklist["Handle Missing"] = True
    
    # 3. REMOVE DUPLICATES
    print("\n✓ STEP 3: Remove Duplicates")
    dup_count = df.duplicated().sum()
    print(f"  - Duplicate rows: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()
    checklist["Remove Duplicates"] = True
    
    # 4. FIX DATA TYPES
    print("\n✓ STEP 4: Fix Data Types")
    print(f"  - Original dtypes:\n{df.dtypes}")
    # Fix dtypes here
    # ...
    checklist["Fix Data Types"] = True
    
    # 5. HANDLE OUTLIERS
    print("\n✓ STEP 5: Handle Outliers")
    numeric_cols = df.select_dtypes(include=['number']).columns
    print(f"  - Numeric columns: {numeric_cols.tolist()}")
    # Detect outliers for each numeric column
    # ...
    checklist["Handle Outliers"] = True
    
    # 6. CLEAN TEXT
    print("\n✓ STEP 6: Clean Text")
    text_cols = df.select_dtypes(include=['object']).columns
    print(f"  - Text columns: {text_cols.tolist()}")
    # Clean text here
    # ...
    checklist["Clean Text"] = True
    
    # 7. FEATURE ENGINEERING
    print("\n✓ STEP 7: Feature Engineering")
    print("  - Creating new features...")
    # Create features here
    # ...
    checklist["Feature Engineering"] = True
    
    # 8. NORMALIZE/SCALE
    print("\n✓ STEP 8: Normalize/Scale")
    print("  - Scaling numeric features...")
    # Scale here
    # ...
    checklist["Normalize/Scale"] = True
    
    # 9. VALIDATE FINAL
    print("\n✓ STEP 9: Validate Final Data")
    print(f"  - Final shape: {df.shape}")
    print(f"  - Final dtypes:\n{df.dtypes}")
    print(f"  - Final missing: {df.isnull().sum().sum()}")
    checklist["Validate Final"] = True
    
    print("\n" + "=" * 60)
    print("CHECKLIST STATUS")
    print("=" * 60)
    for step, done in checklist.items():
        status = "✓" if done else "✗"
        print(f"{status} {step}")
    
    return df
```

---

## 📊 EDA Report Template

Template untuk generate comprehensive EDA report.

```python
def generate_eda_report(df, output_file=None):
    """Generate detailed EDA report"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    report = []
    
    # ===== BASIC INFO =====
    report.append("="*60)
    report.append("EXPLORATORY DATA ANALYSIS REPORT")
    report.append("="*60)
    
    report.append(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # ===== COLUMNS INFO =====
    report.append("\n" + "="*60)
    report.append("COLUMNS INFORMATION")
    report.append("="*60)
    
    for col in df.columns:
        report.append(f"\n{col}:")
        report.append(f"  Type: {df[col].dtype}")
        report.append(f"  Non-null: {df[col].count()}")
        report.append(f"  Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.2f}%)")
        
        if df[col].dtype in ['int64', 'float64']:
            report.append(f"  Mean: {df[col].mean():.2f}")
            report.append(f"  Median: {df[col].median():.2f}")
            report.append(f"  Std: {df[col].std():.2f}")
            report.append(f"  Min: {df[col].min():.2f}")
            report.append(f"  Max: {df[col].max():.2f}")
        else:
            report.append(f"  Unique: {df[col].nunique()}")
            report.append(f"  Top: {df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'}")
    
    # ===== MISSING VALUES =====
    report.append("\n" + "="*60)
    report.append("MISSING VALUES ANALYSIS")
    report.append("="*60)
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col in missing[missing > 0].index:
            report.append(f"\n{col}: {missing[col]} ({missing[col]/len(df)*100:.2f}%)")
    else:
        report.append("\nNo missing values!")
    
    # ===== DUPLICATES =====
    report.append("\n" + "="*60)
    report.append("DUPLICATE ROWS")
    report.append("="*60)
    
    dup = df.duplicated().sum()
    report.append(f"\nTotal duplicates: {dup} ({dup/len(df)*100:.2f}%)")
    
    # ===== STATISTICS =====
    report.append("\n" + "="*60)
    report.append("STATISTICS SUMMARY")
    report.append("="*60)
    report.append(f"\n{df.describe().to_string()}")
    
    # ===== CORRELATION =====
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        report.append("\n" + "="*60)
        report.append("CORRELATION MATRIX")
        report.append("="*60)
        
        corr = df[numeric_cols].corr()
        report.append(f"\n{corr.to_string()}")
    
    # Print and save
    full_report = "\n".join(report)
    print(full_report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_report)
        print(f"\nReport saved to {output_file}")
    
    return full_report
```

---

## 🔄 Complete Cleaning Pipeline

```python
def complete_cleaning_pipeline(df):
    """Complete data cleaning pipeline"""
    
    df_clean = df.copy()
    
    print("Starting data cleaning pipeline...")
    
    # ===== 1. MISSING VALUES =====
    print("\n1. Handling missing values...")
    
    # Drop kolom dengan > 50% missing
    cols_to_drop = df_clean.columns[df_clean.isnull().mean() > 0.5]
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Mean imputation untuk numeric
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    # Mode imputation untuk categorical
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    print(f"  ✓ Missing values handled")
    
    # ===== 2. DUPLICATES =====
    print("\n2. Removing duplicates...")
    
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    
    print(f"  ✓ Removed {before - after} duplicate rows")
    
    # ===== 3. OUTLIERS =====
    print("\n3. Handling outliers...")
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Capping (bukan removal)
        df_clean[col] = df_clean[col].clip(lower, upper)
    
    print(f"  ✓ Outliers capped")
    
    # ===== 4. DATA TYPES =====
    print("\n4. Fixing data types...")
    
    # Convert object dengan numerik value
    for col in cat_cols:
        if pd.to_numeric(df_clean[col], errors='coerce').notna().sum() > len(df_clean) * 0.8:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    print(f"  ✓ Data types fixed")
    
    # ===== 5. STRING CLEANING =====
    print("\n5. Cleaning text...")
    
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].str.strip().str.lower()
    
    print(f"  ✓ Text cleaned")
    
    # ===== 6. VALIDATION =====
    print("\n6. Validating...")
    
    print(f"  - Final shape: {df_clean.shape}")
    print(f"  - Missing values: {df_clean.isnull().sum().sum()}")
    print(f"  - Duplicate rows: {df_clean.duplicated().sum()}")
    print(f"  ✓ Validation passed!")
    
    return df_clean
```

---

## 📊 Data Quality Report

```python
def data_quality_report(df):
    """Generate data quality report"""
    
    print("="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    # 1. Completeness
    print("\n1. COMPLETENESS")
    print("-"*30)
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100
    
    print(f"Total cells: {total_cells}")
    print(f"Missing cells: {missing_cells}")
    print(f"Completeness: {completeness:.2f}%")
    print(f"Quality: {'✓ GOOD' if completeness > 95 else '⚠ FAIR' if completeness > 85 else '✗ POOR'}")
    
    # 2. Uniqueness
    print("\n2. UNIQUENESS (Duplicates)")
    print("-"*30)
    
    dup_rows = df.duplicated().sum()
    uniqueness = (1 - dup_rows / len(df)) * 100
    
    print(f"Total rows: {len(df)}")
    print(f"Duplicate rows: {dup_rows}")
    print(f"Uniqueness: {uniqueness:.2f}%")
    print(f"Quality: {'✓ GOOD' if uniqueness > 98 else '⚠ FAIR' if uniqueness > 95 else '✗ POOR'}")
    
    # 3. Validity (Outliers)
    print("\n3. VALIDITY (Outliers Detection)")
    print("-"*30)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    total_outliers = 0
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        total_outliers += outliers
        
        if outliers > 0:
            outlier_pct = outliers / len(df) * 100
            print(f"  {col}: {outliers} outliers ({outlier_pct:.2f}%)")
    
    validity = (1 - total_outliers / (len(df) * len(numeric_cols))) * 100 if len(numeric_cols) > 0 else 100
    print(f"Overall validity: {validity:.2f}%")
    
    # 4. Consistency
    print("\n4. CONSISTENCY")
    print("-"*30)
    
    # Check data types
    print(f"Data types consistent: ✓")
    print(f"Columns: {df.dtypes.value_counts().to_dict()}")
    
    # 5. Accuracy (Manual check needed)
    print("\n5. ACCURACY")
    print("-"*30)
    print("Manual validation required")
    print("Check ranges, logical relationships, business rules")
    
    # OVERALL SCORE
    print("\n" + "="*60)
    print("OVERALL DATA QUALITY SCORE")
    print("="*60)
    
    score = (completeness + uniqueness + validity) / 3
    
    if score > 95:
        rating = "EXCELLENT ✓"
    elif score > 85:
        rating = "GOOD ⚠"
    elif score > 75:
        rating = "FAIR ⚠"
    else:
        rating = "POOR ✗"
    
    print(f"\nQuality Score: {score:.2f}% - {rating}")
    print(f"  - Completeness: {completeness:.2f}%")
    print(f"  - Uniqueness: {uniqueness:.2f}%")
    print(f"  - Validity: {validity:.2f}%")
```

---

## 🎯 Best Practices Checklist

### Data Loading
- [ ] Check file encoding (UTF-8 recommended)
- [ ] Specify correct delimiter and separator
- [ ] Define missing value indicators
- [ ] Specify column data types if needed

### Data Inspection
- [ ] Check shape (rows × columns)
- [ ] Review column names and types
- [ ] Preview first/last rows
- [ ] Use `.info()` untuk overview lengkap

### Missing Values
- [ ] Identify missing patterns (random vs systematic)
- [ ] Decide strategy per column
- [ ] Document imputation logic
- [ ] Validate results

### Outliers
- [ ] Investigate cause (error vs legitimate)
- [ ] Visualize distribution
- [ ] Decide handling strategy
- [ ] Document decisions

### Data Types
- [ ] Convert to correct types early
- [ ] Handle ambiguous types (date, numeric strings)
- [ ] Use categorical untuk memory efficiency
- [ ] Validate conversions

### Text Data
- [ ] Standardize case and formatting
- [ ] Remove extra whitespace
- [ ] Handle special characters
- [ ] Validate cleanliness

### Feature Engineering
- [ ] Extract meaningful features
- [ ] Document feature logic
- [ ] Validate new features
- [ ] Handle edge cases

---

## 📝 Ringkasan

### EDA & Cleaning Steps

| Step | Purpose | Tools |
| ---- | ------- | ----- |
| 1. Load & Inspect | Understand data | `info()`, `head()`, `describe()` |
| 2. Missing Values | Handle gaps | `fillna()`, `dropna()` |
| 3. Duplicates | Remove copies | `drop_duplicates()` |
| 4. Outliers | Handle extremes | IQR, Z-Score, capping |
| 5. Data Types | Fix types | `astype()`, `to_datetime()` |
| 6. Text Cleaning | Standardize text | `str.strip()`, `str.lower()` |
| 7. Feature Engineering | Create features | `dt`, `str`, custom logic |
| 8. Validation | Quality check | Custom validation functions |

---

## ✏️ Latihan

### Latihan 1: Generate Reports

```python
import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')

# 1. Generate EDA report
generate_eda_report(df, 'eda_report.txt')

# 2. Data quality report
data_quality_report(df)
```

### Latihan 2: Complete Pipeline

```python
# 1. Run complete pipeline
df_clean = complete_cleaning_pipeline(df)

# 2. Verify results
print(df_clean.isnull().sum())
print(df_clean.duplicated().sum())
print(df_clean.dtypes)
```

### Latihan 3: Custom Pipeline

```python
# Create pipeline untuk specific dataset
def my_cleaning_pipeline(df):
    df = df.copy()
    
    # Step 1: ...
    # Step 2: ...
    # Step 3: ...
    
    return df

df_clean = my_cleaning_pipeline(df)
```

---

## 🔗 Referensi

- [Pandas Data Cleaning](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Data Quality Assessment](https://en.wikipedia.org/wiki/Data_quality)
