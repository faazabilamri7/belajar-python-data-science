---
title: Metodologi CRISP-DM
description: Memahami 6 fase CRISP-DM untuk project Data Science
sidebar:
  order: 3
---

## 🔄 CRISP-DM: Metodologi Data Science

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) adalah framework standar untuk project Data Science yang telah terbukti efektif di berbagai industri.

![CRISP-DM Cycle](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/479px-CRISP-DM_Process_Diagram.png)
_Diagram CRISP-DM - Sumber: Wikipedia_

### Mengapa CRISP-DM?

✅ **Proven Framework** - Digunakan oleh perusahaan-perusahaan besar  
✅ **Iterative Process** - Bisa kembali ke fase sebelumnya jika diperlukan  
✅ **Industry-Agnostic** - Bisa diterapkan di berbagai industri  
✅ **Best Practices** - Mengikuti best practices di industry  

---

## 6 Fase CRISP-DM

```
┌─────────────────────────────────────────┐
│                                         │
│    1. Business Understanding            │
│              ↓                          │
│    2. Data Understanding                │
│              ↓                          │
│    3. Data Preparation                  │
│              ↓                          │
│    4. Modeling                          │
│              ↓                          │
│    5. Evaluation                        │
│              ↓                          │
│    6. Deployment                        │
│              ↓                          │
│         (Iterate)                       │
│                                         │
└─────────────────────────────────────────┘
```

---

## Penjelasan Detail Setiap Fase

### 1️⃣ Business Understanding

**Pertanyaan**: Apa masalah bisnis yang ingin diselesaikan?

**Aktivitas:**
- Definisikan objektif bisnis
- Tentukan kriteria sukses
- Identifikasi stakeholders
- Pahami constraints dan resources

**Output:**
- Business problem statement
- Success criteria
- Project plan

**Contoh Studi Kasus - E-Commerce:**
- **Problem**: Tingginya cart abandonment rate (30%)
- **Objective**: Turunkan cart abandonment menjadi 15% dalam 3 bulan
- **Metrics**: Email conversion rate, Revenue per user
- **Stakeholder**: Marketing Manager, Data Scientist, Product Manager

---

### 2️⃣ Data Understanding

**Pertanyaan**: Data apa yang kita punya dan apa kualitasnya?

**Aktivitas:**
- Kumpulkan data yang relevan
- Eksplorasi data (EDA)
- Identifikasi kualitas data
- Cari missing values, outliers, anomali

**Output:**
- Data inventory
- Data quality report
- Preliminary insights

**Contoh:**
```python
import pandas as pd

# Load data
df = pd.read_csv('customer_data.csv')

# Eksplorasi awal
print(f"Shape: {df.shape}")  # 50,000 rows, 25 columns
print(df.info())             # Check types
print(df.describe())         # Summary statistics
print(df.isnull().sum())     # Missing values
```

---

### 3️⃣ Data Preparation

**Pertanyaan**: Bagaimana menyiapkan data untuk modeling?

**Aktivitas:**
- Cleaning data (handle missing values, outliers)
- Feature engineering
- Data transformation
- Data sampling (jika dataset sangat besar)

**Output:**
- Clean dataset
- Feature definitions
- Data dictionary

**Checklist:**
- [ ] Handle missing values (mean, median, delete, interpolate)
- [ ] Fix data types
- [ ] Remove duplicates
- [ ] Handle outliers (IQR, Z-score method)
- [ ] Create new features
- [ ] Scale/normalize numerical features
- [ ] Encode categorical features

**Contoh:**
```python
# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)

# Feature engineering
df['days_since_signup'] = (pd.Timestamp.now() - df['signup_date']).dt.days

# Remove outliers dengan IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]
```

---

### 4️⃣ Modeling

**Pertanyaan**: Model apa yang tepat untuk masalah ini?

**Aktivitas:**
- Pilih algoritma yang sesuai
- Train model
- Tune hyperparameters
- Compare multiple models

**Output:**
- Trained model(s)
- Model parameters
- Training documentation

**Workflow:**

```
Split Data (Train/Validation/Test)
         ↓
   Train Multiple Models
         ↓
   Tune Hyperparameters
         ↓
   Cross-validation
         ↓
   Select Best Model
```

**Contoh Algoritma:**

| Problem | Algoritma | Catatan |
| ------- | --------- | ------- |
| Klasifikasi (Biner) | Logistic Regression, SVM, Random Forest | Mulai dari simple ke complex |
| Klasifikasi (Multi-class) | Decision Tree, Naive Bayes, Gradient Boosting | |
| Regresi | Linear Regression, Ridge/Lasso, Random Forest | |
| Clustering | K-Means, Hierarchical Clustering, DBSCAN | |

---

### 5️⃣ Evaluation

**Pertanyaan**: Seberapa baik model kita?

**Aktivitas:**
- Evaluasi dengan metrik yang tepat
- Validasi dengan business stakeholders
- Compare dengan baseline
- Assess business impact

**Output:**
- Performance metrics
- Model evaluation report
- Go/No-go decision

**Metrik Evaluasi:**

**Classification:**
- Accuracy: Overall correctness
- Precision: Akurasi prediksi positif
- Recall: Ability to find all positives
- F1-Score: Balance antara precision dan recall
- AUC-ROC: Model's ability to distinguish classes

**Regression:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

**Business Metrics:**
- ROI (Return on Investment)
- Revenue impact
- Cost reduction
- Customer satisfaction

---

### 6️⃣ Deployment

**Pertanyaan**: Bagaimana model digunakan di dunia nyata?

**Aktivitas:**
- Deploy ke production
- Setup monitoring
- Create documentation
- Training untuk user

**Output:**
- Production model
- Monitoring dashboard
- Documentation
- Support plan

**Deployment Options:**

| Option | Use Case | Complexity |
| ------ | --------- | ---------- |
| Batch Scoring | Daily/weekly predictions | Low |
| API Service | Real-time predictions | Medium |
| Embedded Model | Mobile/edge deployment | High |
| Dashboard | Interactive insights | Low-Medium |

**Monitoring Metrics:**

- Model accuracy (does performance degrade?)
- Data drift (has input data distribution changed?)
- Prediction latency (is model fast enough?)
- System health (uptime, error rates)

---

## 🔄 Iterasi dalam CRISP-DM

CRISP-DM bukan linear! Sering kita perlu iterate:

```
Business Understanding
    ↓ (Tidak puas dengan hasil)
    ↓ → Kembali ke Data Preparation atau Modeling
    ↓
Data Understanding
    ↓
Data Preparation
    ↓ (Butuh feature baru)
    ↓ → Kembali ke Data Understanding
    ↓
Modeling
    ↓ (Model tidak perform)
    ↓ → Kembali ke Data Preparation
    ↓
Evaluation
    ↓ (Bagus! Proceed)
    ↓
Deployment → Monitoring → Kembali ke Business Understanding (next cycle)
```

---

## 📝 Ringkasan Halaman Ini

### Key Points

| Fase | Fokus | Output |
| ---- | ----- | ------ |
| 1. Business Understanding | Apa masalahnya? | Problem statement |
| 2. Data Understanding | Data apa yang ada? | Data exploration report |
| 3. Data Preparation | Bagaimana siapkan data? | Clean dataset |
| 4. Modeling | Model mana yang terbaik? | Trained model |
| 5. Evaluation | Seberapa bagus? | Performance report |
| 6. Deployment | Bagaimana gunakan? | Production system |

---

## ✏️ Latihan

### Latihan 1: Identifikasi Fase

Untuk setiap skenario berikut, tentukan fase CRISP-DM mana yang sedang berjalan:

1. Tim menganalisis 500,000 transaksi pelanggan untuk pola pembeli
2. Data scientist membandingkan Random Forest vs Gradient Boosting
3. Model sudah live di production, tim monitor accuracy setiap hari
4. Business meminta prediksi revenue untuk Q4 2024
5. Melakukan feature engineering untuk menambah predictive power

### Latihan 2: Studi Kasus

Pilih satu studi kasus di bawah dan jelaskan 6 fase CRISP-DM:

**Case 1: Prediksi Customer Churn**
- Industri: Telecom
- Problem: Terlalu banyak customer yang cancel subscription
- Data: Customer history, usage patterns, support tickets

**Case 2: Fraud Detection**
- Industri: E-commerce
- Problem: Terlalu banyak transaksi fraud
- Data: Transaction history, user behavior, payment info

**Case 3: Product Recommendation**
- Industri: E-commerce
- Problem: Ingin meningkatkan average order value
- Data: Customer purchase history, product info, ratings

---

## 🔗 Referensi

- [CRISP-DM Methodology Official](https://www.datascience-pm.com/crisp-dm-2/)
- [CRISP-DM in Practice - Tutorial](https://www.ibm.com/cloud/learn/crisp-dm)
- [Real-world CRISP-DM Examples](https://towardsdatascience.com/)
