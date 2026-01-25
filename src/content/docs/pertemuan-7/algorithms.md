---
title: ML Algorithms & Scikit-Learn
description: Algoritma machine learning populer dan penggunaan scikit-learn
sidebar:
  order: 3
---

## 🔧 Scikit-Learn: Library ML Python

**Scikit-learn** adalah library Python terpopuler untuk Machine Learning dengan:

- 📦 **Comprehensive** - 100+ algorithms dalam satu library
- 🎯 **Consistent API** - Semua model punya interface yang sama
- 📚 **Well-documented** - Excellent documentation dan examples
- ⚡ **Efficient** - Optimized untuk performance
- 🔌 **Integrates** - Works well dengan pandas, numpy, matplotlib

### Instalasi

```bash
pip install scikit-learn
```

### Import

```python
from sklearn import datasets                    # Sample datasets
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler      # Feature scaling
from sklearn.ensemble import RandomForestClassifier   # Model
from sklearn.metrics import accuracy_score           # Evaluation
```

---

## 🎯 Scikit-Learn API Konsisten

Semua model di scikit-learn mengikuti **3-step workflow yang sama**:

### Template Umum

```python
from sklearn.SomeModule import SomeModel

# STEP 1: Initialize model dengan hyperparameters
model = SomeModel(param1=value1, param2=value2)

# STEP 2: Train pada training data
model.fit(X_train, y_train)

# STEP 3: Predict pada new data
y_pred = model.predict(X_test)

# STEP 4: Evaluate (optional)
score = model.score(X_test, y_test)
```

Konsistensi ini membuat switching antara algorithms sangat mudah!

---

## 📊 Regression Algorithms

### 1. Linear Regression

Model paling sederhana untuk prediksi nilai kontinu.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(np.array([[6]]))
print(f"Prediksi untuk X=6: {y_pred[0]}")  # Output: 12

# Model details
print(f"Slope (m): {model.coef_[0]:.2f}")      # 2.0
print(f"Intercept (b): {model.intercept_:.2f}") # 0.0
```

**Kelebihan:**
- ✅ Sangat sederhana dan fast
- ✅ Interpretable (mudah explain)
- ✅ Baik untuk linear relationships

**Kekurangan:**
- ❌ Hanya untuk linear patterns
- ❌ Sensitive terhadap outliers

### 2. Ridge & Lasso Regression

Linear Regression dengan **regularization** untuk prevent overfitting.

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)  # alpha = regularization strength
ridge.fit(X_train, y_train)
score_ridge = ridge.score(X_test, y_test)

# Lasso (L1 regularization)
lasso = Lasso(alpha=1.0)  # Lasso bisa do feature selection
lasso.fit(X_train, y_train)
score_lasso = lasso.score(X_test, y_test)

print(f"Ridge R²: {score_ridge:.4f}")
print(f"Lasso R²: {score_lasso:.4f}")
```

**Perbedaan:**
- **Ridge**: Penalti pada sum of squares coefficients
- **Lasso**: Penalti pada sum of absolute coefficients (bisa shrink ke 0)

---

## 🎯 Classification Algorithms

### 1. Logistic Regression

Binary classification dengan probabilitas output.

```python
from sklearn.linear_model import LogisticRegression

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)                    # 0 or 1
y_pred_proba = model.predict_proba(X_test)       # [0.2, 0.8]

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

**Kapan menggunakan:**
- ✅ Binary classification
- ✅ Probabilitas output butuh
- ✅ Interpretability penting

### 2. Decision Tree

Membuat keputusan hierarchical dengan aturan if-else.

```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# Train
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Visualize tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, feature_names=feature_names,
               class_names=class_names, rounded=True)
plt.show()

# Feature importance
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

**Parameters:**
- `max_depth` - Kedalaman maksimal tree (prevent overfitting)
- `min_samples_split` - Minimum samples di node untuk split
- `min_samples_leaf` - Minimum samples di leaf node

**Kapan menggunakan:**
- ✅ Interpretability tinggi
- ✅ Mixed data types (numeric + categorical)
- ✅ Non-linear relationships

### 3. Random Forest

Ensemble dari banyak Decision Tree dengan voting.

```python
from sklearn.ensemble import RandomForestClassifier

# Train
model = RandomForestClassifier(
    n_estimators=100,      # Jumlah trees
    max_depth=10,          # Kedalaman setiap tree
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)
```

**Kelebihan:**
- ✅ Robust (less overfitting daripada single tree)
- ✅ Akurasi tinggi
- ✅ Feature importance built-in
- ✅ Handles missing values baik

**Kekurangan:**
- ❌ Less interpretable (black box)
- ❌ Slower prediction time

### 4. K-Nearest Neighbors (KNN)

Klasifikasi berdasarkan K neighbor terdekat.

```python
from sklearn.neighbors import KNeighborsClassifier

# Train (KNN tidak benar-benar "train", hanya store training data)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Score
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

**Parameters:**
- `n_neighbors` (k) - Jumlah neighbors untuk voting
- `metric` - Distance metric ('euclidean', 'manhattan', dll)

**Kapan menggunakan:**
- ✅ Dataset kecil
- ✅ No training phase (lazy learning)
- ✅ Local patterns penting

### 5. Support Vector Machine (SVM)

Find optimal hyperplane pemisah dengan maximum margin.

```python
from sklearn.svm import SVC

# Train
model = SVC(kernel='rbf', C=1.0, gamma='scale')  # rbf kernel
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

**Parameters:**
- `kernel` - 'linear', 'rbf', 'poly' (transformasi data)
- `C` - Regularization strength (higher = less regularization)
- `gamma` - Kernel coefficient

**Kapan menggunakan:**
- ✅ High-dimensional data
- ✅ Binary classification
- ✅ Clear separation margin

---

## 🔀 Clustering Algorithms

### K-Means Clustering

Unsupervised clustering dengan K clusters.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data (unlabeled)
X = np.random.randn(300, 2)

# Train - find K clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Centers
centers = kmeans.cluster_centers_

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, 
            c='red', edgecolors='black', linewidths=2)
plt.title('K-Means Clustering')
plt.show()
```

**Parameters:**
- `n_clusters` - Jumlah clusters
- `init` - Inisialisasi centers ('k-means++' recommended)
- `n_init` - Berapa kali run dengan inisialisasi random

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering

# Train
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = hierarchical.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.show()
```

---

## 🔧 Model Comparison

Berikut contoh training multiple models dan bandingkan:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf')
}

# Train dan evaluate semua
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.4f}")

# Best model
best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} ({results[best_model]:.4f})")
```

---

## ✏️ Practical Example: Complete Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. Feature importance
importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(importance)
```

---

## 📝 Algoritma Cheat Sheet

| Algorithm | Type | Supervised | Use Case |
| --------- | ---- | ---------- | -------- |
| Linear Regression | Regression | Yes | Simple linear prediction |
| Ridge/Lasso | Regression | Yes | Linear + regularization |
| Logistic Regression | Classification | Yes | Binary classification |
| Decision Tree | Classification | Yes | Interpretable decisions |
| Random Forest | Classification | Yes | Robust accuracy |
| KNN | Classification | Yes | Local patterns, small data |
| SVM | Classification | Yes | High-dimensional data |
| K-Means | Clustering | No | Unsupervised grouping |
| PCA | Dimensionality | No | Feature reduction |

---

## 🔗 Referensi

- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Scikit-Learn Algorithms](https://scikit-learn.org/stable/supervised_learning.html)
- [Model Selection Guide](https://scikit-learn.org/stable/modules/ensemble.html)
