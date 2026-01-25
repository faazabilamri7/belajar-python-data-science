---
title: ML Fundamentals
description: Konsep dasar Machine Learning, jenis-jenis, dan workflow
sidebar:
  order: 2
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

## 🎯 Mengapa Machine Learning Penting?

### Real-World Examples

```
1. EMAIL → ML Model → Spam/Not Spam
2. IMAGE → ML Model → Cat/Dog/Bird/...
3. TEXT → ML Model → Sentiment: Positive/Negative
4. MEDICAL DATA → ML Model → Disease/Healthy
5. FINANCIAL DATA → ML Model → Fraud/Legitimate
```

### Keuntungan ML

- 📊 **Scale** - Bisa process jutaan data points
- ⚡ **Speed** - Prediksi instant
- 🧠 **Adaptability** - Improve dengan data baru
- 🔍 **Discovery** - Find patterns yang tidak obvious

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

```python
# Prediksi angka (continuous)
- Prediksi harga rumah: Rp 500 juta, Rp 750 juta, Rp 1 miliar
- Prediksi suhu besok: 25°C, 26.5°C, 28°C
- Prediksi penjualan: 1000 unit, 1500 unit, 2000 unit
```

**Classification** - Prediksi kategori

```python
# Prediksi kategori (discrete)
- Deteksi email: Spam / Not Spam (2 kategori)
- Diagnosis penyakit: Sehat / Flu / COVID / ... (multiple)
- Klasifikasi gambar: Cat / Dog / Bird / ... (multiple)
```

### 2. Unsupervised Learning

Model belajar dari **data tanpa label**. Seperti belajar mandiri menemukan pola.

```
Input → Model → Pattern/Groups
(Features)      (No Labels)

Contoh:
[Customer Data] → Model → Customer Segments
[News Articles] → Model → Article Groups
```

#### Tipe Problem:

**Clustering** - Mengelompokkan data

```python
# Kelompokkan data yang mirip
- Customer segmentation: Bronze, Silver, Gold customers
- Document grouping: Group news by topic
- Anomaly detection: Find unusual patterns
```

**Dimensionality Reduction** - Mengurangi fitur

```python
# Reduce 100 features → 5 features dengan keep penting info
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X)
```

### 3. Reinforcement Learning

Model belajar dari **interaksi dengan environment** melalui reward dan punishment.

```
Agent → Action → Environment → Reward → Learn → Better Action
```

Contoh aplikasi:
- 🎮 Game AI (AlphaGo)
- 🤖 Robotics (Learning to walk)
- 🚗 Self-driving cars

### Perbandingan Ketiga Jenis

| Aspek | Supervised | Unsupervised | Reinforcement |
| ----- | ---------- | ------------ | ------------- |
| **Data** | Labeled | Unlabeled | Environment interaction |
| **Tujuan** | Predict | Find patterns | Maximize reward |
| **Contoh** | Spam detection | Clustering | Game AI |
| **Output** | Prediction | Groups/Patterns | Actions/Policy |
| **Feedback** | Labels | None | Rewards/Penalties |

---

## 🧮 Jenis-Jenis Algoritma

### Supervised Learning Algorithms

#### 1. Linear Regression

Untuk prediksi nilai kontinu dengan **hubungan linear**.

```python
# Konsep: Cari garis terbaik yang fit data
y = mx + b
# harga = koefisien * luas + intercept
```

**Kapan menggunakan:**
- Hubungan input-output linear
- Prediksi harga, sales, temperature

#### 2. Logistic Regression

Untuk **klasifikasi binary** (0 atau 1, Yes/No, Spam/Not Spam).

```python
# Output: probability antara 0 dan 1
P(y=1) = sigmoid(wx + b)
```

**Kapan menggunakan:**
- Klasifikasi 2 kelas
- Email spam/not spam, Customer churn/not churn
- Interpretability penting

#### 3. Decision Tree

Membuat keputusan berdasarkan **aturan if-else yang tersusun hierarchical**.

```
              [Usia > 30?]
              /          \
           Yes            No
            |              |
      [Income > 50K?]   [Student?]
       /       \        /     \
     Yes       No     Yes     No
      |         |      |       |
     Buy    Not Buy   Buy   Not Buy
```

**Kapan menggunakan:**
- Butuh interpretability tinggi (bisa lihat keputusan)
- Data campuran numerik dan kategorikal
- Non-linear relationships

#### 4. Random Forest

**Ensemble** dari banyak Decision Tree yang voting bersama.

```python
# Voting dari 100 trees
For new data:
  Tree 1 → Predict A
  Tree 2 → Predict B
  Tree 3 → Predict A
  ...
  → Majority vote = A
```

**Kapan menggunakan:**
- Akurasi lebih penting dari interpretability
- Mengurangi overfitting
- Feature importance analysis

#### 5. Support Vector Machine (SVM)

Mencari **hyperplane pemisah optimal** antara kelas-kelas.

```python
# Find the best line to separate classes
# dengan maximum margin
```

**Kapan menggunakan:**
- Data dimensi tinggi (many features)
- Binary classification dengan margin jelas
- Small to medium dataset

#### 6. K-Nearest Neighbors (KNN)

Klasifikasi berdasarkan **K tetangga terdekat**.

```python
# Algoritma:
1. Cari K tetangga terdekat
2. Voting mayoritas → kelas
3. Prediksi = class with most neighbors

Contoh: k=5
  - 3 neighbors = Spam
  - 2 neighbors = Not Spam
  → Predict: Spam
```

**Kapan menggunakan:**
- Dataset kecil
- Tidak butuh training (lazy learning)
- Local patterns lebih penting

### Unsupervised Learning Algorithms

#### 1. K-Means Clustering

Mengelompokkan data ke **K cluster** dengan pusat terdekat.

```python
# Algoritma:
1. Pilih K titik pusat random
2. Assign setiap data ke cluster terdekat
3. Update pusat cluster dari mean anggota
4. Repeat sampai konvergen
```

**Kapan menggunakan:**
- Segmentasi customer
- Grouping similar items
- Initial data exploration

#### 2. Hierarchical Clustering

Membuat **hierarki cluster** dari individual ke merged groups.

```
         [All Data]
            /    \
        Group1  Group2
       /      \
      C1      C2
```

#### 3. PCA (Principal Component Analysis)

Mengurangi dimensi dengan **menjaga variance maksimal**.

```python
# Dari 100 features → 10 features
# Keep 95% variance, throw 5%
```

---

## 🔄 Machine Learning Workflow

Berikut adalah workflow standard untuk semua ML projects:

```
┌─────────────────────────────────────────────────────┐
│              ML WORKFLOW (10 STEPS)                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Define Problem                                   │
│         ↓                                            │
│  2. Collect Data                                     │
│         ↓                                            │
│  3. Explore & Analyze (EDA)                          │
│         ↓                                            │
│  4. Clean Data                                       │
│         ↓                                            │
│  5. Feature Engineering                              │
│         ↓                                            │
│  6. Split Data (Train/Test)                          │
│         ↓                                            │
│  7. Train Model                                      │
│         ↓                                            │
│  8. Evaluate Model                                   │
│         ↓                                            │
│  9. Tune Hyperparameters                             │
│         ↓                                            │
│  10. Deploy Model                                    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Step 1: Define Problem

Define masalah apa yang ingin kita selesaikan dengan **jelas dan spesifik**.

```python
# Pertanyaan yang harus dijawab:
# - Apa yang ingin diprediksi? (target variable)
# - Regression atau Classification?
# - Apa metrik keberhasilan?
# - Data apa yang tersedia?

# Contoh:
problem_type = "classification"
target = "churn"  # Apakah customer churn atau tidak
metric = "accuracy"
features_available = ["age", "income", "tenure", "usage"]
```

### Step 2-4: Data Collection, EDA, Cleaning

```python
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Explore
print(df.shape)               # Berapa baris dan kolom
print(df.info())              # Tipe data dan missing values
print(df.describe())          # Statistik: mean, std, min, max
print(df['target'].value_counts())  # Distribusi target

# Clean
df.fillna(df.mean(), inplace=True)  # Handle missing
```

### Step 5: Feature Engineering

Buat fitur baru atau transform fitur yang ada.

```python
# Contoh feature engineering
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                         labels=['Teen', 'Young', 'Middle', 'Senior'])

df['income_per_family'] = df['income'] / df['family_size']

df['is_weekend'] = df['day'].isin(['Saturday', 'Sunday'])
```

### Step 6: Split Data

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)  # Features
y = df['target']                # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training: {X_train.shape}")  # Usually 80%
print(f"Testing: {X_test.shape}")    # Usually 20%
```

### Step 7-8: Train & Evaluate

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

---

## 📏 Konsep Penting

### Train-Test Split

**Mengapa perlu split?**

```
Training Set → Belajar dari data ini
Test Set → Evaluasi pada data baru (belum pernah dilihat)

❌ SALAH: Evaluasi dengan training data
   (akan overly optimistic - bisa 99% accuracy tapi gagal di production)

✅ BENAR: Evaluasi dengan test data
   (realistic measure - tahu seberapa baik generalization)
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
                │      Good Fit         │
                └───────────────────────┘
                   Low ────→ High

Underfitting ← [Sweet Spot] → Overfitting
(Too Simple)   (Just Right)  (Too Complex)
```

| Kondisi | Training Error | Test Error | Solusi |
| ------- | -------------- | ---------- | ------ |
| **Underfitting** | High | High | Model lebih kompleks |
| **Good Fit** | Low | Low | ✅ Optimal |
| **Overfitting** | Very Low | High | Regularization, more data |

### Bias vs Variance

- **High Bias**: Model terlalu simpel, miss patterns (underfitting)
- **High Variance**: Model terlalu sensitif terhadap training data (overfitting)

**Trade-off:**
```
Simple Model (High Bias) ← → Complex Model (High Variance)
```

---

## ✏️ Latihan

### Latihan 1: Konsep

1. Jelaskan perbedaan supervised dan unsupervised learning dengan 2 contoh masing-masing!
2. Kapan menggunakan regression vs classification? Berikan 3 contoh untuk masing-masing.
3. Apa itu overfitting? Bagaimana cara mengatasinya? (Sebutkan minimal 3 cara)

### Latihan 2: Problem Definition

Untuk masing-masing problem, tentukan:
- Supervised atau Unsupervised?
- Regression atau Classification?
- Metrik apa yang digunakan?

**Problems:**
1. Prediksi harga rumah
2. Mendeteksi email spam
3. Segmentasi customer berdasarkan behavior
4. Prediksi penjualan produk
5. Mengenali digit dalam foto

### Latihan 3: Algoritma Selection

Untuk masing-masing scenario, pilih algoritma terbaik dan jelaskan why:

1. Data kecil, butuh interpretability tinggi
2. Large dataset, akurasi maksimal prioritas
3. Binary classification, data tidak terlalu besar
4. Clustering customer untuk marketing campaign

---

## 🔗 Referensi

- [Google ML Crash Course - ML Fundamentals](https://developers.google.com/machine-learning/crash-course)
- [Scikit-Learn Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [Andrew Ng - Machine Learning Specialization](https://www.coursera.org/learn/machine-learning)
