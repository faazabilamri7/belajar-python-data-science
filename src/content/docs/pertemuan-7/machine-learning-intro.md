---
title: Pengenalan Machine Learning
description: Konsep dasar Machine Learning, jenis-jenis algoritma, dan workflow
sidebar:
  order: 1
---

## 🎯 Tujuan Pembelajaran

![Machine Learning](https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=800&h=400&fit=crop)
_Ilustrasi: Machine Learning membuat komputer belajar dari data_

Setelah mempelajari materi ini, kamu akan mampu:

- ✅ Memahami konsep dasar Machine Learning
- ✅ Membedakan jenis-jenis Machine Learning
- ✅ Mengenal algoritma-algoritma populer
- ✅ Memahami workflow Machine Learning

:::tip[Untuk Pengguna Google Colab]
Semua library Machine Learning (scikit-learn, pandas, numpy, matplotlib) sudah terinstall di Google Colab. Kamu bisa langsung praktik tanpa perlu install apapun!

Buka Google Colab di: [colab.research.google.com](https://colab.research.google.com)
:::

---

## 🤖 Apa itu Machine Learning?

**Machine Learning (ML)** adalah cabang dari Artificial Intelligence yang memungkinkan komputer untuk **belajar dari data** tanpa diprogram secara eksplisit.

### Analogi Sederhana

Bayangkan kamu ingin mengajarkan anak kecil mengenali kucing:

**Pendekatan Tradisional (Programming):**

```
IF telinga_runcing AND berkaki_4 AND berekor AND berbulu THEN kucing
```

❌ Masalah: Bagaimana dengan kucing tanpa ekor? Anjing juga berbulu!

**Pendekatan Machine Learning:**

```
Tunjukkan 1000 foto kucing → Komputer menemukan pola sendiri → Bisa mengenali kucing baru
```

✅ Komputer belajar dari contoh, bukan dari aturan yang kaku.

### Definisi Formal

> "A computer program is said to **learn** from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."
> — Tom Mitchell, 1997

- **Experience (E)**: Data yang digunakan untuk training
- **Task (T)**: Apa yang ingin diprediksi/klasifikasi
- **Performance (P)**: Metrik untuk mengukur keberhasilan

---

## 📊 Jenis-Jenis Machine Learning

### 1. Supervised Learning

Model belajar dari **data yang sudah dilabeli**. Seperti belajar dengan guru yang memberikan jawaban benar.

```
Input → Model → Output
(Features)      (Label)

Contoh:
[Luas, Kamar, Lokasi] → Model → Harga Rumah
[Email Text] → Model → Spam / Not Spam
```

#### Tipe Problem:

**Regression** - Prediksi nilai kontinu

- Prediksi harga rumah
- Prediksi suhu besok
- Prediksi penjualan

**Classification** - Prediksi kategori

- Deteksi email spam
- Diagnosis penyakit
- Klasifikasi gambar

### 2. Unsupervised Learning

Model belajar dari **data tanpa label**. Seperti belajar mandiri menemukan pola.

```
Input → Model → Pattern/Groups
(Features)      (No Labels)

Contoh:
[Customer Data] → Model → Customer Segments
```

#### Tipe Problem:

**Clustering** - Mengelompokkan data

- Customer segmentation
- Document grouping
- Anomaly detection

**Dimensionality Reduction** - Mengurangi fitur

- PCA
- t-SNE

### 3. Reinforcement Learning

Model belajar dari **interaksi dengan environment** melalui reward dan punishment.

```
Agent → Action → Environment → Reward → Learn
```

Contoh:

- Game AI (AlphaGo)
- Robotics
- Self-driving cars

### Perbandingan

| Aspek  | Supervised     | Unsupervised  | Reinforcement   |
| ------ | -------------- | ------------- | --------------- |
| Data   | Labeled        | Unlabeled     | Environment     |
| Tujuan | Predict        | Find patterns | Maximize reward |
| Contoh | Spam detection | Clustering    | Game AI         |

![ML Types Comparison](https://miro.medium.com/v2/resize:fit:1400/1*8wU0hfUY3UK_D8Y7tbIyFQ.png)
_Ilustrasi perbandingan jenis-jenis Machine Learning_

---

## 🧮 Algoritma Machine Learning Populer

### Supervised Learning Algorithms

#### 1. Linear Regression

Untuk prediksi nilai kontinu dengan hubungan linear.

```python
# Contoh konsep
y = mx + b
# harga = koefisien * luas + intercept
```

**Kapan menggunakan:**

- Hubungan input-output linear
- Prediksi harga, sales, dll

#### 2. Logistic Regression

Untuk klasifikasi binary (0 atau 1).

```python
# Probability output
P(y=1) = sigmoid(wx + b)
```

**Kapan menggunakan:**

- Klasifikasi 2 kelas
- Spam/not spam, churn/not churn

#### 3. Decision Tree

Membuat keputusan berdasarkan aturan if-else.

```
                 [Usia > 30?]
                 /          \
              Yes            No
               |              |
        [Income > 50K?]    [Student?]
         /       \          /     \
      Yes        No       Yes      No
        |         |         |        |
      Buy      Not Buy    Buy    Not Buy
```

**Kapan menggunakan:**

- Butuh interpretability tinggi
- Data campuran numerik dan kategorikal

#### 4. Random Forest

Kumpulan banyak Decision Tree (Ensemble).

**Kapan menggunakan:**

- Akurasi lebih penting dari interpretability
- Mengurangi overfitting

#### 5. K-Nearest Neighbors (KNN)

Klasifikasi berdasarkan tetangga terdekat.

```
Untuk data baru:
1. Cari K tetangga terdekat
2. Voting mayoritas → kelas
```

**Kapan menggunakan:**

- Dataset kecil
- Tidak butuh training

#### 6. Support Vector Machine (SVM)

Mencari hyperplane pemisah optimal.

**Kapan menggunakan:**

- Data dimensi tinggi
- Klasifikasi dengan margin jelas

#### 7. Neural Networks / Deep Learning

Terinspirasi dari otak manusia.

**Kapan menggunakan:**

- Data sangat besar
- Image, Text, Audio

### Unsupervised Learning Algorithms

#### 1. K-Means Clustering

Mengelompokkan data ke K cluster.

```python
# Algoritma:
1. Pilih K titik pusat random
2. Assign setiap data ke cluster terdekat
3. Update pusat cluster
4. Repeat sampai konvergen
```

#### 2. Hierarchical Clustering

Membuat hierarki cluster.

#### 3. PCA (Principal Component Analysis)

Mengurangi dimensi dengan menjaga variance.

---

## 🔄 Machine Learning Workflow

```
┌─────────────────────────────────────────────────────┐
│                   ML WORKFLOW                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Define Problem                                   │
│         ↓                                            │
│  2. Collect Data                                     │
│         ↓                                            │
│  3. Explore & Prepare Data (EDA + Cleaning)          │
│         ↓                                            │
│  4. Feature Engineering                              │
│         ↓                                            │
│  5. Split Data (Train/Test)                          │
│         ↓                                            │
│  6. Train Model                                      │
│         ↓                                            │
│  7. Evaluate Model                                   │
│         ↓                                            │
│  8. Tune Hyperparameters                             │
│         ↓                                            │
│  9. Deploy Model                                     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 1. Define Problem

Langkah pertama adalah clearly define masalah apa yang ingin kita selesaikan. Ini termasuk menentukan target variable (apa yang ingin diprediksi), jenis problem (regression atau classification), dan metrik apa yang akan digunakan untuk mengukur sukses.

```python
# Pertanyaan yang harus dijawab:
# - Apa yang ingin diprediksi? (target variable)
# - Regression atau Classification?
# - Apa metrik keberhasilan?
# - Data apa yang tersedia?

problem_type = "classification"  # atau "regression"
target = "churn"  # kolom yang ingin diprediksi - apakah customer churn atau tidak
metric = "accuracy"  # atau "rmse" untuk regression
```

### 2. Collect & Explore Data

Setelah define problem, kita perlu load data dan explore untuk memahami struktur dan kualitas data. Gunakan functions seperti `head()`, `info()`, `describe()` untuk lihat gambaran awal dataset sebelum preprocessing.

```python
import pandas as pd

# Load data - bisa dari CSV, database, atau API
df = pd.read_csv('data.csv')

# Explore - pahami struktur dan karakteristik data
print(df.shape)               # Berapa baris dan kolom
print(df.info())              # Info tipe data dan missing values
print(df.describe())          # Statistik deskriptif (mean, std, min, max, dll)
print(df['target'].value_counts())  # Distribusi target variable
```

### 3. Prepare Data

Sebelum training model, data perlu di-prepare: handle missing values, encode categorical variables, dan scale numeric features. Ini sangat penting karena model ML membutuhkan input yang bersih dan normalized.

```python
# Handle missing values - isi dengan mean dari kolom tersebut
df.fillna(df.mean(), inplace=True)

# Encode categorical - convert kategori teks jadi numerical (one-hot encoding)
df = pd.get_dummies(df, columns=['category_column'])

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

### 4. Split Data

Split data menjadi training set (untuk melatih model) dan test set (untuk evaluasi). Ini penting agar kita bisa mengukur performance pada data yang belum pernah dilihat model sebelumnya (generalization).

```python
from sklearn.model_selection import train_test_split

# Pisahkan features (X) dan target (y)
X = df.drop('target', axis=1)  # Features - variabel yang digunakan untuk prediksi
y = df['target']                # Target - variabel yang ingin diprediksi

# Split: 80% training, 20% testing - standard split ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # random_state untuk reproducibility
)

print(f"Training set: {X_train.shape}")  # Jumlah training data
print(f"Test set: {X_test.shape}")        # Jumlah test data
```

### 5. Train Model

Langkah ini adalah di mana model "belajar" dari training data. Model akan adjust parameternya untuk meminimalkan prediction error pada training set.

```python
from sklearn.ensemble import RandomForestClassifier

# Inisialisasi model - set hyperparameters
model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees dalam random forest

# Training - model belajar dari training data
model.fit(X_train, y_train)  # Fit model ke training data
```

### 6. Evaluate Model

Setelah model dilatih, evaluasi performance-nya pada test set (data yang belum pernah dilihat). Gunakan berbagai metrik untuk mendapatkan gambaran lengkap tentang seberapa baik model bekerja.

```python
from sklearn.metrics import accuracy_score, classification_report

# Prediksi - gunakan trained model untuk prediksi pada test data
y_pred = model.predict(X_test)

# Evaluasi - ukur seberapa baik prediksi kita
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # Persentase prediksi yang benar
print(classification_report(y_test, y_pred))  # Report detail: precision, recall, F1-score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

---

## 📏 Konsep Penting

### Train-Test Split

**Mengapa perlu split?**

- Training data: untuk melatih model
- Testing data: untuk mengevaluasi (data yang belum pernah dilihat model)

```python
# JANGAN PERNAH evaluasi dengan training data!
# Ini akan memberikan hasil yang terlalu optimis (overfitting)

# BENAR:
y_pred = model.predict(X_test)  # Evaluasi dengan data baru
```

### Overfitting vs Underfitting

```
                    Model Complexity
                ┌───────────────────────┐
                │                       │
   Error        │    \      Optimal     │
     ↑          │     \        ↓        │
                │      \      ╱         │
                │       \    ╱          │
                │        \  ╱           │
                │         ╲╱            │
                │                       │
                └───────────────────────┘
                   Low ────────────→ High

     Underfitting ←──── Sweet Spot ────→ Overfitting
```

| Kondisi      | Training Error | Test Error | Solusi                    |
| ------------ | -------------- | ---------- | ------------------------- |
| Underfitting | High           | High       | Model lebih kompleks      |
| Good Fit     | Low            | Low        | ✅ Optimal                |
| Overfitting  | Very Low       | High       | Regularization, more data |

### Bias vs Variance

- **High Bias**: Model terlalu simpel (underfitting)
- **High Variance**: Model terlalu sensitif terhadap training data (overfitting)

**Trade-off:** Semakin kompleks model, variance naik tapi bias turun.

### Feature Engineering

Proses membuat fitur baru dari fitur yang ada.

```python
# Contoh feature engineering
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                         labels=['Teen', 'Young', 'Middle', 'Senior'])

df['income_per_family'] = df['income'] / df['family_size']

df['is_weekend'] = df['day'].isin(['Saturday', 'Sunday'])
```

---

## 🔧 Scikit-Learn: Library ML Python

### API Konsisten

Semua model di scikit-learn memiliki interface yang sama:

```python
from sklearn.some_module import SomeModel

# 1. Inisialisasi
model = SomeModel(hyperparameters)

# 2. Training
model.fit(X_train, y_train)

# 3. Prediksi
y_pred = model.predict(X_test)

# 4. Evaluasi (opsional - bisa pakai function terpisah)
score = model.score(X_test, y_test)
```

### Contoh Berbagai Model

Sekarang mari kita lihat berbagai algoritma machine learning yang umum digunakan dan kode untuk melatihnya. Setiap algoritma memiliki kelebihan dan kekurangan, dan pilihan algoritma tergantung pada problem type dan karakteristik data:

```python
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Semua menggunakan API yang sama!
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: {score:.4f}")
```

---

## 📝 Praktik: Klasifikasi Sederhana

Sekarang mari kita praktik dengan membuat model klasifikasi sederhana. Kita akan menggunakan dataset Iris yang populer dan melakukan supervised learning end-to-end. Mari kita praktik:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data (contoh: Iris dataset)
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

---

## 📝 Ringkasan

### ML Types

| Type          | Data        | Tujuan          | Contoh         |
| ------------- | ----------- | --------------- | -------------- |
| Supervised    | Labeled     | Predict         | Spam detection |
| Unsupervised  | Unlabeled   | Find patterns   | Clustering     |
| Reinforcement | Environment | Maximize reward | Game AI        |

### Supervised Learning

| Problem        | Output     | Contoh           |
| -------------- | ---------- | ---------------- |
| Regression     | Continuous | Price prediction |
| Classification | Categories | Spam/Not Spam    |

### ML Workflow

1. Define Problem
2. Collect Data
3. EDA & Cleaning
4. Feature Engineering
5. Train-Test Split
6. Train Model
7. Evaluate
8. Tune
9. Deploy

---

## ✏️ Latihan

### Latihan 1: Konsep

1. Jelaskan perbedaan supervised dan unsupervised learning dengan contoh!
2. Kapan menggunakan regression vs classification?
3. Apa itu overfitting dan bagaimana cara mengatasinya?

### Latihan 2: Praktik

1. Load dataset Titanic dari seaborn
2. Identifikasi: ini problem classification atau regression?
3. Tentukan feature (X) dan target (y)
4. Split data menjadi train dan test

### Latihan 3: Model Comparison

1. Train 3 model berbeda pada dataset yang sama
2. Bandingkan akurasi masing-masing
3. Mana model terbaik? Mengapa?

---

## ❓ FAQ (Pertanyaan yang Sering Diajukan)

### Q: Berapa banyak data yang butuh untuk Machine Learning?

**A:** Tergantung complexity model:

- **Simple model** (Linear Regression): 100-1000 data cukup
- **Complex model** (Deep Learning): Butuh 10,000-1,000,000+ data

**Rule of thumb**: Min 10x jumlah features. Contoh: 10 features = min 100 data.

Tapi kualitas lebih penting dari kuantitas!

### Q: Bagaimana cara tahu jika model overfitting?

**A:** Cek gap antara training error dan test error:

- Training accuracy: 95%, Test accuracy: 70% → **OVERFITTING**
- Training accuracy: 85%, Test accuracy: 82% → **GOOD FIT**

Solusi: Tambah data, regularization, atau simplify model.

### Q: Saya punya imbalanced data (contoh: 95% class A, 5% class B)

**A:** Ini problem serius karena model akan bias ke class A. Solusi:

1. **Resampling** - Oversample minority atau undersample majority
2. **Stratified Split** - Jaga ratio di train dan test
3. **Class Weight** - Berikan weight lebih ke minority class
4. **Different Metric** - Jangan pakai accuracy, gunakan F1-score atau AUC

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y  # Jaga ratio
)
```

### Q: Feature Engineering itu seni atau sains?

**A:** **Keduanya!** Tapi mostly **seni**:

- **Sains** - Ada best practices (scaling, encoding, etc)
- **Seni** - Bergantung domain knowledge dan kreativitas

Contoh domain knowledge: Untuk prediksi penjualan, jam 10 pagi vs 10 malam berbeda. Kamu harus tahu ini!

### Q: Bagaimana cara pilih model?

**A:** Sederhanakan dulu:

1. Start dengan **simple model** (Linear Regression, Logistic Regression)
2. Kalibrasi baseline → **Random Forest** (robust, less tuning)
3. Kalau perlu akurasi ultra tinggi → **Neural Networks**

**Ocam's Razor**: Model sederhana yang akurat > model kompleks yang akurat sedikit lebih tinggi.

### Q: Apa itu hyperparameter tuning?

**A:** Hyperparameter adalah setting model yang tidak belajar dari data, harus set manual:

```python
model = RandomForest(
    n_estimators=100,  # ← hyperparameter (berapakah yg optimal?)
    max_depth=10,      # ← hyperparameter
    min_samples_split=2  # ← hyperparameter
)
```

Cara find optimal:

1. **Grid Search** - Coba semua kombinasi
2. **Random Search** - Coba kombinasi random
3. **Bayesian Optimization** - Smart search

### Q: Accuracy 90% itu bagus atau buruk?

**A:** **Tergantung:**

- Prediksi penyakit: 90% bisa kurang (false positive harmful)
- Rekomendasi produk: 90% sudah OK (false positive tidak serious)

**Better approach**: Lihat precision, recall, F1-score, tidak hanya accuracy!

### Q: Bagaimana cara prevent overfitting?

**A:** Beberapa strategi:

1. **More data** - Model akan less "creative"
2. **Regularization** - Penalti untuk kompleksitas (L1, L2)
3. **Early Stopping** - Stop training sebelum overfit
4. **Cross-validation** - Better estimate of test performance
5. **Simpler model** - Kurangi features atau model complexity

---

:::tip[Pro Tip]
Machine Learning itu iteratif. Jarang sekali model pertama langsung bagus. Expect untuk iterate banyak kali, trying different approaches!
