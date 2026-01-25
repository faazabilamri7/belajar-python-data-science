---
title: ML Workflow & End-to-End Pipeline
description: Complete machine learning workflow dari load data sampai evaluasi
sidebar:
  order: 2
---

## 📊 ML Workflow Overview

Berikut adalah workflow standard yang digunakan di semua ML projects:

```
1. Load Data
     ↓
2. Explore & Analyze (EDA)
     ↓
3. Data Cleaning
     ↓
4. Preprocessing & Feature Engineering
     ↓
5. Train-Test Split
     ↓
6. Train Model
     ↓
7. Evaluate Model
     ↓
8. Hyperparameter Tuning (Optional)
     ↓
9. Final Prediction
```

---

## Step 1-2: Load & Explore Data

### Load Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Opsi 1: CSV file
df = pd.read_csv('data.csv')

# Opsi 2: Built-in dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Opsi 3: Seaborn dataset
df = sns.load_dataset('titanic')

print(f"Shape: {df.shape}")  # Rows x Columns
print(f"\nFirst 5 rows:")
print(df.head())
```

### Explore Dataset

```python
# Info dasar
print(df.info())          # Data types, non-null counts
print(df.describe())      # Statistical summary
print(df.isnull().sum())  # Missing values

# Cek target variable
if 'target' in df.columns:
    print("\nTarget Distribution:")
    print(df['target'].value_counts())
```

### Visualize Data

```python
# Untuk classification - target distribution
plt.figure(figsize=(10, 5))
df['target'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), 
            annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Feature distributions
df.select_dtypes(include=[np.number]).hist(figsize=(15, 10))
plt.tight_layout()
plt.show()
```

---

## Step 3-4: Data Cleaning & Preprocessing

### Handle Missing Values

```python
# Cek missing
print(df.isnull().sum())

# Strategy 1: Drop rows dengan missing
df_clean = df.dropna()

# Strategy 2: Fill dengan mean (numeric)
df['age'] = df['age'].fillna(df['age'].mean())

# Strategy 3: Fill dengan mode (categorical)
df['category'] = df['category'].fillna(df['category'].mode()[0])

# Strategy 4: Drop columns dengan banyak missing
df = df.drop(columns=['col_with_90_missing'])
```

### Encode Categorical Variables

```python
# One-Hot Encoding
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Manual mapping
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
```

### Feature Engineering

```python
# Create new features
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                         labels=['child', 'young', 'middle', 'senior'])

df['bmi'] = df['weight'] / (df['height'] ** 2)

df['is_weekend'] = df['day'].isin(['Saturday', 'Sunday']).astype(int)

# One-hot encode new features
df = pd.get_dummies(df, columns=['age_group'])
```

---

## Step 5: Train-Test Split

### Basic Split

```python
from sklearn.model_selection import train_test_split

# Pisahkan features dan target
X = df.drop('target', axis=1)  # Features
y = df['target']                # Target

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,          # 20% test
    random_state=42,        # For reproducibility
    stratify=y              # Keep class distribution (classification only)
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Fit scaler pada training data ONLY
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply same transformation ke test data
X_test_scaled = scaler.transform(X_test)

print(f"Train mean: {X_train_scaled.mean():.4f}")
print(f"Train std: {X_train_scaled.std():.4f}")
```

---

## Step 6: Train Model

### Training Single Model

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize
model = RandomForestClassifier(
    n_estimators=100,       # Jumlah trees
    max_depth=10,           # Kedalaman maksimal
    random_state=42
)

# Train
model.fit(X_train_scaled, y_train)

print("✅ Model training complete!")
```

### Train Multiple Models for Comparison

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC()
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"✅ {name} trained")
```

---

## Step 7: Evaluate Model

### Make Predictions

```python
# Predict pada test set
y_pred = model.predict(X_test_scaled)

# Predict probabilities (jika ada)
y_pred_proba = model.predict_proba(X_test_scaled)
```

### Basic Metrics

```python
from sklearn.metrics import accuracy_score, classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Visualize Results

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### Compare Multiple Models

```python
# Evaluate semua model
results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.4f}")

# Best model
best_model = max(results, key=results.get)
print(f"\n🏆 Best model: {best_model} ({results[best_model]:.4f})")
```

---

## 🔄 Complete End-to-End Example: Titanic

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("="*50)
print("COMPLETE ML PIPELINE: TITANIC SURVIVAL PREDICTION")
print("="*50)

# 1. LOAD DATA
print("\n1️⃣ Loading data...")
df = sns.load_dataset('titanic')
print(f"   Shape: {df.shape}")

# 2. EXPLORE
print("\n2️⃣ Exploring data...")
print(df.info())
print(f"   Missing values: {df.isnull().sum().sum()}")

# 3. CLEAN & PREPROCESS
print("\n3️⃣ Preprocessing...")

# Handle missing
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Select features
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
X = df[features].copy()
y = df['survived'].copy()

# Encode categorical
X['sex'] = X['sex'].map({'male': 1, 'female': 0})
X = pd.get_dummies(X, columns=['embarked'], drop_first=True)
X = X.fillna(X.mean())

print(f"   Features: {X.columns.tolist()}")
print(f"   Target: {y.value_counts().to_dict()}")

# 4. SPLIT
print("\n4️⃣ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 5. SCALE
print("\n5️⃣ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. TRAIN
print("\n6️⃣ Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. EVALUATE
print("\n7️⃣ Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# Feature importance
print("\n📊 Feature Importance:")
for feat, imp in sorted(zip(X.columns, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"   {feat}: {imp:.4f}")

print("\n" + "="*50)
print("✅ PIPELINE COMPLETE!")
print("="*50)
```

---

## 💾 Save & Load Model

```python
import joblib

# Save model
joblib.dump(model, 'titanic_model.pkl')
joblib.dump(scaler, 'titanic_scaler.pkl')
print("✅ Model saved!")

# Load model
loaded_model = joblib.load('titanic_model.pkl')
loaded_scaler = joblib.load('titanic_scaler.pkl')
print("✅ Model loaded!")

# Use loaded model
X_new_scaled = loaded_scaler.transform(X_new)
predictions = loaded_model.predict(X_new_scaled)
```

---

## 🎯 Best Practices Checklist

### Before Training
- [ ] Data shape correct?
- [ ] No missing values (or handled)?
- [ ] Categorical variables encoded?
- [ ] Features scaled?
- [ ] Train-test split done?

### During Training
- [ ] Model training (no errors)?
- [ ] Training time reasonable?
- [ ] Loss/metrics decreasing?

### After Training
- [ ] Evaluated on test set?
- [ ] Metrics make sense?
- [ ] Predictions look reasonable?
- [ ] Model saved (if needed)?

---

## ✏️ Latihan

### Latihan 1: Iris Classification

```python
from sklearn.datasets import load_iris

# 1. Load iris dataset
# 2. Split into train/test
# 3. Scale features
# 4. Train RandomForestClassifier
# 5. Evaluate on test set
# 6. Print accuracy & classification report
```

### Latihan 2: Titanic with Your Own Preprocessing

```python
# Load titanic dataset
# Do your own data cleaning & preprocessing
# Select features you think are important
# Train & evaluate model
# Compare your accuracy with baseline
```

### Latihan 3: Multi-Model Comparison

```python
# Train 5 different models on same dataset
# Compare accuracy on test set
# Create visualization comparing results
# Identify best model
```

---

## 🔗 Referensi

- [Scikit-Learn Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [Preprocessing Methods](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
