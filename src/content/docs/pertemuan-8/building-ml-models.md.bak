---
title: Membangun & Mengevaluasi Model ML
description: Praktik membangun model Machine Learning dan mengukur kinerjanya
sidebar:
  order: 1
---

## 🎯 Tujuan Pembelajaran

![Building ML Models](https://images.unsplash.com/photo-1518932945647-7a1c969f8be2?w=800&h=400&fit=crop)
_Ilustrasi: Membangun model ML adalah puncak dari perjalanan Data Science_

Setelah mempelajari materi ini, kamu akan mampu:

- ✅ Membangun model ML end-to-end
- ✅ Memahami berbagai metrik evaluasi
- ✅ Melakukan cross-validation
- ✅ Melakukan hyperparameter tuning
- ✅ Menginterpretasi hasil model

---

## 💻 Setup Google Colab

:::tip[Mengapa Google Colab?]
Google Colab adalah platform GRATIS dari Google yang menyediakan:

- 🆓 GPU/TPU gratis untuk training model
- ☁️ Tidak perlu install apapun di laptop
- 💾 Langsung terhubung dengan Google Drive
- 🔧 Library ML sudah terinstall (pandas, sklearn, dll)

**Cocok untuk laptop dengan spesifikasi rendah!**
:::

### Cara Menggunakan Google Colab

1. **Buka Google Colab**: Kunjungi [colab.research.google.com](https://colab.research.google.com)
2. **Login** dengan akun Google kamu
3. **Buat Notebook Baru**: Klik `File` → `New Notebook`
4. **Mulai Coding!** Ketik kode di cell dan tekan `Shift + Enter` untuk run

### Tips Penting di Colab

Sebelum mulai coding, cek apakah kamu sudah di Google Colab environment. Di Colab, hampir semua library ML sudah pre-installed, jadi tidak perlu install manual.

```python
# Cek apakah kamu di Google Colab
import sys
IN_COLAB = 'google.colab' in sys.modules
print(f"Running in Colab: {IN_COLAB}")

# Jika di Colab, semua library ML sudah terinstall!
# Tidak perlu pip install pandas, numpy, sklearn, dll
```

---

## 📊 Dataset untuk Latihan

:::note[Dataset yang Akan Kita Gunakan]
Kita akan menggunakan **Titanic Dataset** dari Kaggle - dataset paling populer untuk belajar ML!

**Link Kaggle:** [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

Dataset ini berisi data penumpang Titanic dan kita akan memprediksi siapa yang selamat (Survived).
:::

### Cara Load Dataset di Colab (MUDAH!)

Ada 2 cara mudah untuk load Titanic dataset:

**Cara 1: Langsung dari Seaborn (Paling Mudah!)**

Cara paling mudah adalah load dataset langsung dari Seaborn, yang sudah menyediakan berbagai dataset populer untuk belajar ML. Tidak perlu download atau upload file apapun!

```python
import seaborn as sns
import pandas as pd

# Load Titanic dataset langsung - tidak perlu download!
df = sns.load_dataset('titanic')
print(df.head())
print(f"\nJumlah data: {len(df)} baris")
```

**Cara 2: Download dari Kaggle (Opsional)**

```python
# Jika ingin data langsung dari Kaggle:
# 1. Download dari: https://www.kaggle.com/c/titanic/data
# 2. Upload file train.csv ke Colab (klik icon folder di kiri)
# 3. Load dengan:
# df = pd.read_csv('train.csv')
```

---

## �🔄 Review: ML Pipeline

```python
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
df = pd.read_csv('data.csv')

# 2. Prepare Features (X) and Target (y)
X = df.drop('target', axis=1)
y = df['target']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Predict & Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 📊 Metrik Evaluasi: Classification

### Confusion Matrix

Tabel yang menunjukkan prediksi vs actual.

```
                    Predicted
                 Positive  Negative
Actual Positive    TP        FN
       Negative    FP        TN
```

| Istilah                 | Penjelasan                                          |
| ----------------------- | --------------------------------------------------- |
| **True Positive (TP)**  | Prediksi positif, actual positif ✅                 |
| **True Negative (TN)**  | Prediksi negatif, actual negatif ✅                 |
| **False Positive (FP)** | Prediksi positif, actual negatif ❌ (Type I Error)  |
| **False Negative (FN)** | Prediksi negatif, actual positif ❌ (Type II Error) |

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Hitung confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualisasi
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negatif', 'Positif'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

### Accuracy

Proporsi prediksi yang benar dari total prediksi.

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

Accuracy adalah metrik paling sederhana - menunjukkan persentase prediksi yang benar. Namun, metrik ini tidak selalu reliable untuk dataset yang imbalanced. Mari kita hitung accuracy untuk model kita:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

**Kapan menggunakan?**

- ✅ Dataset balanced
- ❌ Tidak cocok untuk imbalanced dataset

### Precision

Dari semua prediksi positif, berapa yang benar-benar positif?

$$Precision = \frac{TP}{TP + FP}$$

Precision mengukur seberapa akurat prediksi positif kita. Precision tinggi berarti ketika model memprediksi "positif", biasanya benar. Mari kita hitung precision:

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")
```

**Kapan menggunakan?**

- ✅ Ketika False Positive costly
- Contoh: Spam detection (jangan sampai email penting masuk spam)

### Recall (Sensitivity)

Dari semua actual positif, berapa yang berhasil diprediksi positif?

$$Recall = \frac{TP}{TP + FN}$$

Recall mengukur seberapa banyak dari semua kasus positif yang berhasil terdeteksi. Recall tinggi berarti model tidak melewatkan banyak positive cases. Mari kita hitung recall:

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}")
```

**Kapan menggunakan?**

- ✅ Ketika False Negative costly
- Contoh: Deteksi penyakit (jangan sampai orang sakit terdeteksi sehat)

### F1-Score

Harmonic mean dari Precision dan Recall.

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

F1-Score adalah metrik yang bagus ketika kita ingin balance antara precision dan recall, terutama untuk dataset yang imbalanced. Mari kita hitung F1-Score:

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")
```

**Kapan menggunakan?**

- ✅ Dataset imbalanced
- ✅ Butuh balance antara precision dan recall

### Classification Report

Laporan klasifikasi (Classification Report) menyajikan semua metrik penting (Precision, Recall, F1-Score) dalam satu tabel yang rapi dan mudah dibaca. Ini sangat berguna untuk mendapatkan overview menyeluruh tentang performa model kita:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
```

Output:

```
              precision    recall  f1-score   support

     Class 0       0.85      0.90      0.87       100
     Class 1       0.88      0.82      0.85        90

    accuracy                           0.86       190
   macro avg       0.86      0.86      0.86       190
weighted avg       0.86      0.86      0.86       190
```

### ROC Curve & AUC

**ROC (Receiver Operating Characteristic)** curve menunjukkan trade-off antara True Positive Rate dan False Positive Rate.

**AUC (Area Under Curve)** mengukur kemampuan model membedakan kelas.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Untuk model yang bisa output probability
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Hitung ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

| AUC     | Interpretasi |
| ------- | ------------ |
| 0.5     | Random guess |
| 0.7-0.8 | Acceptable   |
| 0.8-0.9 | Excellent    |
| > 0.9   | Outstanding  |

---

## 📈 Metrik Evaluasi: Regression

### Mean Absolute Error (MAE)

Rata-rata absolute error.

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

MAE adalah metrik regression yang mengukur rata-rata selisih absolut antara prediksi dan nilai actual. MAE mudah diinterpretasi karena satuan sama dengan target variable. Semakin kecil MAE, semakin akurat prediksi model:

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")
```

**Interpretasi:** Rata-rata prediksi meleset sebesar MAE unit.

### Mean Squared Error (MSE)

Rata-rata squared error.

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

MSE adalah metrik regression yang menghitung rata-rata kuadrat error. Dengan mengkuadratkan error, MSE memberikan penalti lebih berat untuk error yang besar. Ini berguna ketika error besar lebih problematic daripada error kecil:

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")
```

### Root Mean Squared Error (RMSE)

Akar dari MSE, satuan sama dengan target.

$$RMSE = \sqrt{MSE}$$

RMSE adalah versi "ditarik akar" dari MSE, sehingga satuan dan interpretasinya sama dengan target variable kita. RMSE sering digunakan karena lebih intuitive dan menunjukkan rata-rata error dalam unit yang original. Mari kita hitung RMSE:

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")
```

### R² Score (Coefficient of Determination)

Proporsi variance yang dijelaskan oleh model.

R² Score mengukur seberapa baik model menjelaskan variasi dalam target variable. R² berkisar dari 0 sampai 1, dimana 1 berarti perfect fit. R² juga sering diinterpretasi sebagai "berapa persen variance target yang dijelaskan oleh model". Mari kita hitung R² Score:

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
```

| R²  | Interpretasi               |
| --- | -------------------------- |
| 1.0 | Perfect fit                |
| 0.0 | Same as predicting mean    |
| < 0 | Worse than predicting mean |

---

## 🔀 Cross-Validation

### Mengapa Cross-Validation?

Single train-test split bisa memberikan hasil yang bervariasi. Cross-validation memberikan estimasi yang lebih reliable.

### K-Fold Cross-Validation

K-Fold Cross-Validation adalah teknik yang membagi data menjadi k bagian, kemudian melakukan training dan testing sebanyak k kali dengan fold yang berbeda setiap kali. Ini memberikan estimasi performa yang lebih reliable dan stabil dibanding single train-test split.

Berikut ilustrasi bagaimana K-Fold bekerja:

```
Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Fold 1: Train=[3-10], Test=[1-2]
Fold 2: Train=[1,2,5-10], Test=[3-4]
Fold 3: Train=[1-4,7-10], Test=[5-6]
Fold 4: Train=[1-6,9-10], Test=[7-8]
Fold 5: Train=[1-8], Test=[9-10]

Final Score = Average of 5 folds
```

Sekarang mari kita implementasi K-Fold Cross-Validation dalam kode:

```python
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(random_state=42)

# 5-fold cross validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
```

### Stratified K-Fold

Untuk dataset yang imbalanced (misalnya 90% class 0 dan 10% class 1), regular K-Fold bisa menghasilkan fold yang tidak representative. Stratified K-Fold mengatasi ini dengan mempertahankan proporsi kelas di setiap fold, sehingga setiap fold adalah representative dari keseluruhan dataset:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

### Cross-Validation untuk Multiple Metrics

Kadang kita ingin mengevaluasi model menggunakan multiple metrics sekaligus (accuracy, precision, recall, f1). Fungsi `cross_validate` memungkinkan kita melakukan ini dengan lebih efficient. Mari kita evaluasi model dengan multiple metrics:

```python
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 🎛️ Hyperparameter Tuning

### Apa itu Hyperparameter?

**Parameters**: Dipelajari dari data (weights, coefficients)
**Hyperparameters**: Diset sebelum training (learning rate, n_estimators)

### Grid Search

Grid Search adalah teknik brute-force untuk menemukan hyperparameter terbaik dengan mencoba semua kombinasi hyperparameter yang kita specify. Meskipun computational expensive, Grid Search dapat menemukan kombinasi optimal dengan pasti. Mari kita gunakan GridSearchCV untuk mencari hyperparameter terbaik:

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

### Random Search

Random Search adalah alternatif yang lebih efisien untuk Grid Search, terutama ketika hyperparameter space sangat besar. Daripada mencoba semua kombinasi, Random Search mencoba kombinasi random dengan jumlah yang kita specify. Ini sering menemukan kombinasi yang baik dengan computational cost yang jauh lebih rendah. Mari kita gunakan RandomizedSearchCV:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Random Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,  # Jumlah kombinasi yang dicoba
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.4f}")
```

---

## 📊 Feature Importance

### Tree-based Models

Untuk tree-based models seperti Decision Tree, Random Forest, Gradient Boosting, kita bisa mengakses feature importance langsung dari model. Feature importance menunjukkan seberapa penting setiap feature dalam membuat prediksi. Feature dengan importance tinggi berkontribusi lebih banyak terhadap prediksi model. Mari kita hitung dan visualisasi feature importance:

```python
import matplotlib.pyplot as plt

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
features = X.columns

# Plot
plt.figure(figsize=(10, 6))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

### Permutation Importance

Permutation Importance adalah model-agnostic technique yang bisa digunakan untuk semua jenis model (tidak hanya tree-based). Teknik ini mengevaluasi pentingnya feature dengan melihat seberapa banyak performa model turun ketika nilai feature di-shuffle (diacak). Feature yang lebih penting akan menyebabkan performa turun lebih signifikan. Mari kita hitung Permutation Importance:

```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Plot
plt.figure(figsize=(10, 6))
sorted_idx = perm_importance.importances_mean.argsort()[::-1]
plt.bar(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
plt.xticks(range(len(sorted_idx)), [features[i] for i in sorted_idx], rotation=45, ha='right')
plt.title('Permutation Importance')
plt.tight_layout()
plt.show()
```

---

## 🏗️ End-to-End Project: Prediksi Survivor Titanic

:::tip[Project Pemula-Friendly]
Kita akan menggunakan **Titanic Dataset** - dataset paling populer untuk belajar ML!

- 🆓 Gratis & mudah diakses
- 📊 Ukuran kecil (891 baris) - cocok untuk laptop apapun
- 🎯 Problem klasifikasi sederhana: Survived (1) atau Not Survived (0)
  :::

### Langkah 1: Setup & Load Data

Langkah pertama adalah mengimport semua library yang kita butuhkan. Di Google Colab, semua library populer seperti scikit-learn, pandas, numpy, matplotlib sudah terinstall, jadi kita tidak perlu melakukan instalasi. Mari kita import semua library yang diperlukan:

```python
# ============================================
# STEP 1: IMPORT LIBRARY & LOAD DATA
# ============================================
# Jalankan di Google Colab - semua library sudah terinstall!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Library untuk Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Setting agar grafik lebih bagus
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✅ Semua library berhasil di-import!")
```

### Langkah 2: Load Dataset Titanic

Sekarang kita akan load Titanic dataset dari Seaborn. Dataset ini berisi informasi tentang 891 penumpang Titanic, dan kita akan memprediksi siapa yang selamat berdasarkan fitur seperti age, gender, class, etc. Mari kita load dataset dan lihat struktur datanya:

```python
# ============================================
# STEP 2: LOAD TITANIC DATASET
# ============================================
# Cara paling mudah: langsung dari seaborn!

df = sns.load_dataset('titanic')

# Lihat 5 baris pertama
print("📊 Data Titanic:")
print(df.head())

print(f"\n📏 Ukuran dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"\n📋 Kolom-kolom yang tersedia:")
print(df.columns.tolist())
```

### Langkah 3: Exploratory Data Analysis (EDA)

Sebelum membuat model, kita harus memahami data kita terlebih dahulu. Langkah ini disebut Exploratory Data Analysis (EDA). Kita akan melihat tipe data, missing values, dan distribusi target variable. Mari kita lakukan EDA:

```python
# ============================================
# STEP 3: EXPLORASI DATA
# ============================================

# Cek info dataset
print("📋 Info Dataset:")
print(df.info())

print("\n" + "="*50)
print("🎯 TARGET: Kolom 'survived'")
print("="*50)
print(df['survived'].value_counts())
print(f"\nPersentase yang selamat: {df['survived'].mean()*100:.1f}%")
```

```python
# Visualisasi: Siapa yang selamat?
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Berdasarkan Gender
sns.countplot(data=df, x='sex', hue='survived', ax=axes[0])
axes[0].set_title('Survival berdasarkan Gender')
axes[0].legend(['Tidak Selamat', 'Selamat'])

# 2. Berdasarkan Kelas Tiket
sns.countplot(data=df, x='pclass', hue='survived', ax=axes[1])
axes[1].set_title('Survival berdasarkan Kelas Tiket')
axes[1].legend(['Tidak Selamat', 'Selamat'])

# 3. Distribusi Umur
sns.histplot(data=df, x='age', hue='survived', kde=True, ax=axes[2])
axes[2].set_title('Distribusi Umur')

plt.tight_layout()
plt.show()
```

### Langkah 4: Data Cleaning & Preprocessing

Data yang kita dapat dari dunia nyata tidak selalu sempurna - ada missing values, data yang tidak konsisten, dan kolom yang tidak relevan. Langkah Data Cleaning adalah tahap krusial untuk memastikan data kita berkualitas sebelum training model. Mari kita clean dan preprocess data:

```python
# ============================================
# STEP 4: DATA CLEANING
# ============================================

# Cek missing values
print("🔍 Missing Values:")
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")
```

```python
# Pilih kolom yang akan kita gunakan (yang paling penting)
# Kita pilih fitur yang mudah dipahami pemula

kolom_penting = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df_clean = df[kolom_penting].copy()

print("📋 Kolom yang kita gunakan:")
for col in kolom_penting:
    print(f"  - {col}")
```

```python
# Handle missing values dengan cara sederhana

# 1. Isi missing age dengan median (nilai tengah)
median_age = df_clean['age'].median()
df_clean['age'] = df_clean['age'].fillna(median_age)
print(f"✅ Missing 'age' diisi dengan median: {median_age}")

# 2. Isi missing embarked dengan mode (nilai paling sering)
mode_embarked = df_clean['embarked'].mode()[0]
df_clean['embarked'] = df_clean['embarked'].fillna(mode_embarked)
print(f"✅ Missing 'embarked' diisi dengan mode: {mode_embarked}")

# Cek lagi
print(f"\n🔍 Missing values sekarang: {df_clean.isnull().sum().sum()}")
```

```python
# Encode categorical variables (ubah text jadi angka)

# Sex: male=1, female=0
df_clean['sex'] = df_clean['sex'].map({'male': 1, 'female': 0})

# Embarked: S=0, C=1, Q=2
df_clean['embarked'] = df_clean['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("✅ Categorical encoding selesai!")
print(df_clean.head())
```

### Langkah 5: Siapkan Data untuk Model

Sebelum training model, kita harus memisahkan features (X) dan target (y), lalu split data menjadi training dan testing set. Training set digunakan untuk melatih model, sedangkan testing set digunakan untuk evaluasi performa model yang belum pernah dilihat sebelumnya. Mari kita siapkan data:

```python
# ============================================
# STEP 5: PERSIAPAN DATA
# ============================================

# Pisahkan Features (X) dan Target (y)
X = df_clean.drop('survived', axis=1)  # Semua kolom kecuali 'survived'
y = df_clean['survived']               # Hanya kolom 'survived'

print(f"📊 Features (X): {X.shape}")
print(f"🎯 Target (y): {y.shape}")
print(f"\n📋 Fitur yang digunakan: {X.columns.tolist()}")
```

```python
# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% untuk testing
    random_state=42,    # Agar hasil reproducible
    stratify=y          # Menjaga proporsi kelas
)

print(f"📊 Training set: {X_train.shape[0]} data")
print(f"📊 Test set: {X_test.shape[0]} data")
```

```python
# Scaling fitur (standardization)
# Ini penting agar semua fitur punya skala yang sama

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling selesai!")
```

### Langkah 6: Training Model Pertama (Baseline)

Untuk mengawali, kita akan training model paling sederhana: Logistic Regression. Model ini akan menjadi baseline kita untuk membandingkan dengan model yang lebih complex. Baseline penting karena memberikan referensi minimum yang harus dilampaui oleh model lain. Mari kita train model baseline:

```python
# ============================================
# STEP 6: BASELINE MODEL
# ============================================
print("="*50)
print("🚀 BASELINE MODEL: Logistic Regression")
print("="*50)

# Buat dan train model
model_baseline = LogisticRegression(random_state=42)
model_baseline.fit(X_train_scaled, y_train)

# Prediksi
y_pred_baseline = model_baseline.predict(X_test_scaled)

# Evaluasi
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
print(f"\n✅ Accuracy: {accuracy_baseline:.4f} ({accuracy_baseline*100:.1f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_baseline,
                          target_names=['Tidak Selamat', 'Selamat']))
```

### Langkah 7: Bandingkan Beberapa Model

Sekarang kita akan mencoba beberapa model berbeda dan membandingkan performa mereka. Dengan membandingkan multiple models, kita dapat menemukan model mana yang paling cocok untuk problem kita. Kita akan menggunakan 5-Fold Cross Validation untuk mendapatkan estimasi performa yang lebih reliable. Mari kita bandingkan:

```python
# ============================================
# STEP 7: BANDINGKAN MODEL
# ============================================
print("="*50)
print("🔄 PERBANDINGAN MODEL (5-Fold Cross Validation)")
print("="*50)

# Daftar model yang akan kita coba
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Simpan hasil
results = {}

for name, model in models.items():
    # Cross validation dengan 5 fold
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    results[name] = scores

    print(f"\n📊 {name}:")
    print(f"   Accuracy per fold: {[f'{s:.3f}' for s in scores]}")
    print(f"   Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Visualisasi perbandingan
plt.figure(figsize=(10, 5))
plt.boxplot(results.values(), labels=results.keys())
plt.title('Perbandingan Accuracy Model (5-Fold CV)')
plt.ylabel('Accuracy')
plt.show()
```

### Langkah 8: Training Model Terbaik

Berdasarkan perbandingan di langkah sebelumnya, kita sudah mengetahui model mana yang memiliki performa terbaik. Sekarang kita akan training model terbaik (Random Forest) dengan data training lengkap, dan kemudian evaluasi dengan data testing:

```python
# ============================================
# STEP 8: TRAIN MODEL TERBAIK
# ============================================
print("="*50)
print("🏆 TRAINING MODEL TERBAIK: Random Forest")
print("="*50)

# Train Random Forest dengan data training
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Prediksi di test set
y_pred_final = best_model.predict(X_test_scaled)

# Evaluasi final
accuracy_final = accuracy_score(y_test, y_pred_final)
print(f"\n✅ Final Accuracy: {accuracy_final:.4f} ({accuracy_final*100:.1f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_final,
                          target_names=['Tidak Selamat', 'Selamat']))
```

### Langkah 9: Visualisasi Hasil

Untuk memahami hasil prediksi model lebih baik, kita perlu membuat visualisasi. Confusion Matrix menunjukkan jenis kesalahan yang dilakukan model, sedangkan Feature Importance menunjukkan fitur mana yang paling berpengaruh dalam prediksi. Mari kita visualisasi hasil model:

```python
# ============================================
# STEP 9: VISUALISASI HASIL
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Tidak Selamat', 'Selamat'],
            yticklabels=['Tidak Selamat', 'Selamat'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Actual')

# 2. Feature Importance
feature_names = X.columns
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

axes[1].barh(range(len(importances)), importances[indices])
axes[1].set_yticks(range(len(importances)))
axes[1].set_yticklabels([feature_names[i] for i in indices])
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance')

plt.tight_layout()
plt.show()

# Penjelasan Feature Importance
print("\n📊 Penjelasan Feature Importance:")
for i in indices:
    print(f"  {feature_names[i]}: {importances[i]:.4f}")
```

### Langkah 10: Coba Prediksi Data Baru

Sekarang yang paling menyenangkan - menggunakan model yang sudah dilatih untuk memprediksi data baru! Kita akan membuat beberapa contoh penumpang baru dan melihat apakah model memprediksi mereka selamat atau tidak. Kita juga akan melihat confidence (probabilitas) dari setiap prediksi:

```python
# ============================================
# STEP 10: PREDIKSI PENUMPANG BARU
# ============================================
print("="*50)
print("🔮 PREDIKSI PENUMPANG BARU")
print("="*50)

# Contoh penumpang baru
# Format: [pclass, sex, age, sibsp, parch, fare, embarked]
# sex: 0=female, 1=male
# embarked: 0=Southampton, 1=Cherbourg, 2=Queenstown

penumpang_baru = [
    [3, 1, 25, 0, 0, 7.25, 0],    # Pria kelas 3, umur 25, sendiri
    [1, 0, 35, 1, 0, 100, 1],     # Wanita kelas 1, umur 35, dengan pasangan
    [2, 0, 28, 0, 2, 30, 0],      # Wanita kelas 2, umur 28, dengan 2 anak
]

deskripsi = [
    "Pria kelas 3, umur 25, sendiri",
    "Wanita kelas 1, umur 35, dengan pasangan",
    "Wanita kelas 2, umur 28, dengan 2 anak"
]

# Scale data baru
penumpang_scaled = scaler.transform(penumpang_baru)

# Prediksi
predictions = best_model.predict(penumpang_scaled)
probabilities = best_model.predict_proba(penumpang_scaled)

print("\nHasil Prediksi:")
for i, (desc, pred, prob) in enumerate(zip(deskripsi, predictions, probabilities)):
    hasil = "🟢 SELAMAT" if pred == 1 else "🔴 TIDAK SELAMAT"
    print(f"\n{i+1}. {desc}")
    print(f"   Prediksi: {hasil}")
    print(f"   Probabilitas: {prob[1]*100:.1f}% selamat")
```

---

## 💾 Menyimpan Model (Opsional)

Setelah training model dan mendapatkan hasil yang baik, kita mungkin ingin menyimpan model untuk digunakan kemudian. Dengan joblib, kita bisa menyimpan model dan semua preprocessing objects (seperti scaler) dalam file pickle. Mari kita simpan model:

```python
# Simpan model jika ingin digunakan nanti
import joblib

# Simpan model dan scaler
joblib.dump(best_model, 'titanic_model.pkl')
joblib.dump(scaler, 'titanic_scaler.pkl')

print("✅ Model berhasil disimpan!")

# Cara load kembali:
# loaded_model = joblib.load('titanic_model.pkl')
# loaded_scaler = joblib.load('titanic_scaler.pkl')
```

:::tip[Download Model dari Colab]
Untuk download model yang sudah disimpan:

```python
from google.colab import files
files.download('titanic_model.pkl')
files.download('titanic_scaler.pkl')
```

:::

---

## 📝 Ringkasan

### Talking Points Hari Ini

| Topik                                   | Penjelasan                                           |
| --------------------------------------- | ---------------------------------------------------- |
| Implementasi Scikit-Learn               | `from sklearn import ...`, API yang konsisten (fit, predict, score) |
| Evaluasi Matriks (Accuracy, Precision, Recall, MSE) | Accuracy = keseluruhan benar, Precision = tepat positif, Recall = tangkap semua positif |
| Interpretasi Model                      | Feature importance, coefficients, decision boundaries |
| Deployment Sederhana (Concept)          | Simpan model dengan joblib/pickle, load untuk prediksi |

### Classification Metrics

| Metric    | Rumus          | Kapan Gunakan   |
| --------- | -------------- | --------------- |
| Accuracy  | (TP+TN)/Total  | Balanced data   |
| Precision | TP/(TP+FP)     | FP costly       |
| Recall    | TP/(TP+FN)     | FN costly       |
| F1        | 2×P×R/(P+R)    | Imbalanced data |

### Regression Metrics

| Metric | Interpretasi              |
| ------ | ------------------------- |
| MAE    | Average absolute error    |
| MSE    | Average squared error     |
| RMSE   | √MSE, same unit as target |
| R²     | Variance explained        |

---

## ✏️ Latihan Final Project

### Project: Pilih Salah Satu Dataset Berikut

:::note[Dataset Ramah Pemula dari Kaggle]
Semua dataset ini GRATIS dan mudah diakses. Pilih yang paling menarik untuk kamu!
:::

#### 1. 🚢 Titanic - Survival Prediction (Paling Mudah!)

**Link:** [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

**Tujuan:** Prediksi penumpang selamat atau tidak

**Cara Load:**

```python
import seaborn as sns
df = sns.load_dataset('titanic')
```

---

#### 2. 🐧 Palmer Penguins (Alternatif Iris, Lebih Modern!)

**Link:** [kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data)

**Tujuan:** Klasifikasi spesies penguin

**Cara Load:**

```python
import seaborn as sns
df = sns.load_dataset('penguins')
```

---

#### 3. 🌸 Iris Flower Classification (Klasik!)

**Link:** [kaggle.com/datasets/uciml/iris](https://www.kaggle.com/datasets/uciml/iris)

**Tujuan:** Klasifikasi jenis bunga iris

**Cara Load:**

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
```

---

#### 4. 💳 Credit Card Default (Lebih Menantang)

**Link:** [kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

**Tujuan:** Prediksi apakah nasabah akan gagal bayar

**Cara Load di Colab:**

```python
# Download dari Kaggle, upload ke Colab, lalu:
df = pd.read_csv('UCI_Credit_Card.csv')
```

---

#### 5. 🏠 California Housing (Regression)

**Link:** [kaggle.com/datasets/camnugent/california-housing-prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

**Tujuan:** Prediksi harga rumah

**Cara Load:**

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['price'] = housing.target
```

---

### Langkah Pengerjaan Final Project

#### Step 1: Setup Notebook di Colab

```python
# Jalankan ini di cell pertama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("✅ Setup selesai! Siap untuk Final Project!")
```

#### Step 2: Template Final Project

```python
# ============================================
# FINAL PROJECT: [Nama Dataset Kamu]
# Nama: [Nama Kamu]
# ============================================

# 1. LOAD DATA
# ------------
# ... load dataset pilihan kamu ...

# 2. EXPLORATORY DATA ANALYSIS
# ----------------------------
# - Tampilkan df.info()
# - Cek missing values
# - Buat minimal 3 visualisasi
# - Tulis insight yang kamu temukan

# 3. DATA PREPROCESSING
# ---------------------
# - Handle missing values
# - Encode categorical variables
# - Split train/test
# - Feature scaling

# 4. MODELING
# -----------
# - Train minimal 3 model berbeda
# - Gunakan cross-validation
# - Bandingkan hasilnya

# 5. EVALUATION
# -------------
# - Classification report
# - Confusion matrix
# - Feature importance

# 6. KESIMPULAN
# -------------
# Tulis kesimpulan dan insight kamu!
```

### Kriteria Penilaian

| Aspek            | Bobot | Penjelasan                               |
| ---------------- | ----- | ---------------------------------------- |
| 📊 EDA           | 20%   | Minimal 3 visualisasi + insight          |
| 🧹 Preprocessing | 20%   | Handle missing values, encoding, scaling |
| 🤖 Modeling      | 25%   | Minimal 3 model + cross-validation       |
| 📈 Evaluasi      | 25%   | Metrics lengkap + interpretasi           |
| 📝 Dokumentasi   | 10%   | Notebook rapi + penjelasan jelas         |

### Tips Sukses Final Project

:::tip[Tips dari Mentor]

1. **Mulai dari yang sederhana** - Pilih Titanic atau Penguins untuk project pertama
2. **Jangan takut error** - Error adalah bagian dari proses belajar!
3. **Dokumentasi setiap langkah** - Tulis penjelasan di setiap cell
4. **Tanya jika bingung** - Mentor dan teman-teman siap membantu
5. **Enjoy the process!** 🎉
   :::

---

## ❓ FAQ (Pertanyaan yang Sering Diajakan)

### Q: Saya dapat akurasi 99% di training tapi 60% di testing, apa masalahnya?

**A:** Classic **OVERFITTING**! Model menghafal data training, tapi tidak bisa generalize. Solusi:

1. Kurangi kompleksitas model (fewer features, simpler algorithm)
2. Tambah data training
3. Gunakan regularization (L1, L2)
4. Increase test size atau gunakan cross-validation

Ini adalah masalah paling common di ML!

### Q: Bagaimana cara handle imbalanced dataset?

**A:** Dataset imbalanced (contoh: 95% survived, 5% dead) adalah problem serius. Solusi:

1. **Use stratified split**: Jaga ratio di train-test
2. **Resampling**: Oversample minority atau undersample majority
3. **Class weights**: Berikan weight lebih ke minority class
4. **Better metrics**: Gunakan F1, AUC, precision-recall instead of accuracy

```python
# Stratified split
from sklearn.model_selection import train_test_split
train_test_split(X, y, test_size=0.2, stratify=y)

# Class weight di model
model = RandomForestClassifier(class_weight='balanced')
```

### Q: Apa itu confusion matrix dan bagaimana interpretasinya?

**A:** Confusion matrix menunjukkan benar/salah prediksi:

```
                    Predicted
                Positive  Negative
Actual Positive    TP        FN
       Negative    FP        TN

- TP (True Positive): Prediksi positif, benar positif ✅
- FN (False Negative): Prediksi negatif, tapi seharusnya positif ❌
- FP (False Positive): Prediksi positif, tapi seharusnya negatif ❌
- TN (True Negative): Prediksi negatif, benar negatif ✅
```

**Rumus penting:**

- Accuracy = (TP+TN)/(TP+TN+FP+FN)
- Precision = TP/(TP+FP) - Dari prediksi positif, berapa yg benar?
- Recall = TP/(TP+FN) - Dari positif sebenarnya, berapa yg tertangkap?

### Q: Harus pakai Precision atau Recall?

**A:** Tergantung cost:

- **High Precision important** (minimize FP): Spam detection (lebih baik miss spam than block legit email)
- **High Recall important** (minimize FN): Disease detection (lebih baik false alarm than miss disease)

**F1-score** adalah harmonic mean keduanya untuk balance.

### Q: Cross-validation itu apa gunanya?

**A:** Cross-validation digunakan untuk:

1. **Better estimate** of true model performance (tidak biased ke 1 train-test split)
2. **Reduce variance** dalam performa estimate
3. **Use all data** untuk training dan validation

**Contoh K-Fold (K=5):**

```
Data → [Fold1, Fold2, Fold3, Fold4, Fold5]
        Train on 4 folds, test on 1 fold (5x)
        Average hasil 5x → Better estimate!
```

### Q: Hyperparameter tuning dengan GridSearchCV vs RandomizedSearchCV?

**A:**

- **GridSearchCV** - Coba semua kombinasi hyperparameter (lengkap tapi lambat)
- **RandomizedSearchCV** - Coba kombinasi random (cepat tapi less thorough)

**Rekomendasi:**

- Dataset kecil/few hyperparameters → GridSearchCV
- Dataset besar/many hyperparameters → RandomizedSearchCV

### Q: Feature importance itu apa dan bagaimana interpretasinya?

**A:** Feature importance menunjukkan kontribusi setiap feature terhadap prediksi. Contoh:

```python
# Untuk tree-based model
importances = model.feature_importances_
features = X.columns
for feat, imp in sorted(zip(features, importances),
                         key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.4f}")

# Output:
# Age: 0.2456 ← Age paling penting!
# Sex: 0.1890
# Fare: 0.1234
# Pclass: 0.0987
```

Gunakan ini untuk:

1. **Feature selection** - Drop low importance features
2. **Business insight** - Apa driver paling penting?

### Q: Saya training model tapi loss tidak turun, ada masalah?

**A:** Beberapa kemungkinan:

1. **Learning rate terlalu tinggi** → Loss bounces around
2. **Learning rate terlalu rendah** → Learning sangat lambat
3. **Data bukan normalize** → Model struggling
4. **Bad hyperparameters** → Wrong algorithm untuk problem ini

**Debugging:**

1. Cek apakah target variable benar-benar berbeda
2. Plot loss curve - patterns menunjukkan masalah apa
3. Start dengan simple model untuk baseline
4. Cek data quality - jangan ada NaN atau outliers ekstrem

### Q: Berapa ukuran test set yang ideal?

**A:** Umumnya:

- **80-20 split** - Standard (80% train, 20% test)
- **70-30 split** - Saat data kecil
- **90-10 split** - Saat data sangat besar

Tapi yang lebih penting adalah gunakan **cross-validation** untuk lebih robust estimate!

---

:::note[Catatan Penting]
Pertemuan ini adalah puncak dari perjalanan belajar kamu. Kamu sudah belajar dari fundamental Python sampai Building ML Models!

Ingat: **Machine Learning bukan destination, tapi journey.** Terus belajar, experiment, dan practice. Setiap model yang kamu build adalah pembelajaran.

Good luck! 🚀
:::

**Butuh bantuan?** Jangan ragu untuk bertanya di grup atau sesi mentoring!
:::
