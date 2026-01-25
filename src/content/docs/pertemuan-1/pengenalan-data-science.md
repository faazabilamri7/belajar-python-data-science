---
title: Pengenalan Data Science
description: Memahami apa itu Data Science, peran Data Scientist, dan siklus CRISP-DM
sidebar:
  order: 1
---

## 🎯 Tujuan Pembelajaran

Setelah mempelajari materi ini, kamu akan mampu:

- ✅ Memahami apa itu Data Science dan mengapa penting
- ✅ Mengenal berbagai peran dalam ekosistem data
- ✅ Memahami metodologi CRISP-DM
- ✅ Mengetahui tools dan teknologi yang digunakan

---

## 🤔 Apa itu Data Science?

![Data Science Illustration](https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=400&fit=crop)
_Ilustrasi: Data Science mengubah data mentah menjadi insights berharga_

**Data Science** adalah bidang multidisiplin yang menggunakan metode ilmiah, algoritma, dan sistem untuk mengekstrak pengetahuan dan insights dari data terstruktur maupun tidak terstruktur.

### Analogi Sederhana

Bayangkan kamu adalah **detektif** 🔍:

- **Data** = Bukti-bukti yang dikumpulkan
- **Data Science** = Proses investigasi untuk menemukan kebenaran
- **Insights** = Kesimpulan dari investigasi

### Mengapa Data Science Penting?

Di era digital, data dihasilkan setiap detik:

- 📱 500 juta tweets per hari
- 📧 300 miliar email per hari
- 🎥 500 jam video di-upload ke YouTube per menit
- 💳 Jutaan transaksi finansial per detik

**Data Science** membantu organisasi:

1. 📈 **Membuat keputusan berbasis data** (Data-Driven Decision)
2. 🔮 **Memprediksi tren masa depan** (Predictive Analytics)
3. 🤖 **Mengotomasi proses** (Automation)
4. 💡 **Menemukan peluang baru** (Business Intelligence)

---

## 👥 Peran dalam Ekosistem Data

### 1. Data Analyst

- **Fokus**: Menganalisis data historis
- **Output**: Report, Dashboard, Insights
- **Tools**: Excel, SQL, Tableau, Power BI

### 2. Data Engineer

- **Fokus**: Membangun infrastruktur data
- **Output**: Data Pipeline, Data Warehouse
- **Tools**: Python, SQL, Spark, Airflow

### 3. Data Scientist

- **Fokus**: Membangun model prediktif
- **Output**: ML Models, Predictions
- **Tools**: Python, R, TensorFlow, Scikit-learn

### 4. Machine Learning Engineer

- **Fokus**: Deploy dan maintain ML models
- **Output**: Production-ready ML systems
- **Tools**: Python, Docker, Kubernetes, MLflow

### Perbandingan

| Aspek       | Data Analyst        | Data Scientist           | Data Engineer              |
| ----------- | ------------------- | ------------------------ | -------------------------- |
| Pertanyaan  | "Apa yang terjadi?" | "Apa yang akan terjadi?" | "Bagaimana data mengalir?" |
| Skill Utama | SQL, Visualisasi    | Statistics, ML           | Programming, Cloud         |
| Output      | Reports             | Models                   | Pipelines                  |

---

## 🔄 CRISP-DM: Metodologi Data Science

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) adalah framework standar untuk project Data Science.

![CRISP-DM Cycle](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/479px-CRISP-DM_Process_Diagram.png)
_Diagram CRISP-DM - Sumber: Wikipedia_

### 6 Fase CRISP-DM

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

### Penjelasan Setiap Fase

#### 1️⃣ Business Understanding

**Pertanyaan**: Apa masalah bisnis yang ingin diselesaikan?

- Definisikan objektif bisnis
- Tentukan kriteria sukses
- Identifikasi stakeholders

**Contoh**: _"Bagaimana cara mengurangi customer churn sebesar 20%?"_

#### 2️⃣ Data Understanding

**Pertanyaan**: Data apa yang kita punya?

- Kumpulkan data yang relevan
- Eksplorasi data (EDA)
- Identifikasi kualitas data

**Contoh**: _"Kita punya data transaksi pelanggan 2 tahun terakhir"_

#### 3️⃣ Data Preparation

**Pertanyaan**: Bagaimana menyiapkan data untuk modeling?

- Cleaning data (handle missing values, outliers)
- Feature engineering
- Data transformation

**Contoh**: _"Handle 5% missing values di kolom income"_

#### 4️⃣ Modeling

**Pertanyaan**: Model apa yang tepat untuk masalah ini?

- Pilih algoritma yang sesuai
- Train model
- Tune hyperparameters

**Contoh**: _"Gunakan Random Forest untuk prediksi churn"_

#### 5️⃣ Evaluation

**Pertanyaan**: Seberapa baik model kita?

- Evaluasi dengan metrik yang tepat
- Validasi dengan business stakeholders
- Compare dengan baseline

**Contoh**: _"Model mencapai accuracy 85% dan precision 78%"_

#### 6️⃣ Deployment

**Pertanyaan**: Bagaimana model digunakan di dunia nyata?

- Deploy ke production
- Monitor performance
- Maintenance dan update

**Contoh**: _"Deploy sebagai API untuk sistem CRM"_

---

## 🛠️ Tools & Teknologi Data Science

### Bahasa Pemrograman

| Bahasa    | Kegunaan             | Popularitas |
| --------- | -------------------- | ----------- |
| 🐍 Python | General purpose, ML  | ⭐⭐⭐⭐⭐  |
| 📊 R      | Statistical analysis | ⭐⭐⭐⭐    |
| 🗄️ SQL    | Database queries     | ⭐⭐⭐⭐⭐  |

### Python Libraries

```python
# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```

### Environment & Tools

- **Jupyter Notebook** - Interactive coding
- **Google Colab** - Free GPU/TPU
- **VS Code** - Code editor
- **Git** - Version control

---

## 📊 Contoh Aplikasi Data Science

### 1. E-Commerce

- 🛒 Product recommendation
- 📦 Demand forecasting
- 💰 Dynamic pricing

### 2. Healthcare

- 🏥 Disease prediction
- 💊 Drug discovery
- 📋 Patient analytics

### 3. Finance

- 💳 Fraud detection
- 📈 Stock prediction
- 🏦 Credit scoring

### 4. Transportation

- 🚗 Route optimization
- 🚌 Demand prediction
- 🚦 Traffic analysis

---

## 🧠 Skill yang Dibutuhkan

### Hard Skills

1. **Programming** - Python, SQL
2. **Mathematics** - Linear Algebra, Calculus
3. **Statistics** - Probability, Hypothesis Testing
4. **Machine Learning** - Algorithms, Evaluation

### Soft Skills

1. **Problem Solving** - Analytical thinking
2. **Communication** - Explain complex to simple
3. **Curiosity** - Always asking "why?"
4. **Business Acumen** - Understand business context

---

## 📝 Ringkasan

### Talking Points Hari Ini

| Topik                       | Penjelasan                                                                 |
| --------------------------- | -------------------------------------------------------------------------- |
| Definisi Data Science, AI & ML | Data Science mengekstrak insights dari data, AI membuat mesin cerdas, ML melatih model dari data |
| Karir di Bidang Data        | Data Analyst, Data Engineer, Data Scientist, ML Engineer                   |
| Workflow Proyek Data        | CRISP-DM: Business Understanding → Data Understanding → Preparation → Modeling → Evaluation → Deployment |
| Etika Data dan Privasi      | Penggunaan data yang bertanggung jawab, menjaga privasi pengguna           |

---

## ✏️ Latihan

### Pertanyaan Refleksi

1. Sebutkan 3 contoh penerapan Data Science dalam kehidupan sehari-harimu!
2. Fase CRISP-DM mana yang menurutmu paling challenging? Mengapa?
3. Skill apa yang sudah kamu miliki dan mana yang perlu dikembangkan?

### Mini Project

Cari 1 case study Data Science dari perusahaan favoritmu dan identifikasi:

- Masalah bisnis apa yang diselesaikan?
- Data apa yang digunakan?
- Hasil/impact apa yang dicapai?

---

## 🔗 Referensi Tambahan

- [What is Data Science? - IBM](https://www.ibm.com/topics/data-science)
- [CRISP-DM Methodology](https://www.datascience-pm.com/crisp-dm-2/)
- [Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

---

## ❓ FAQ (Pertanyaan yang Sering Diajukan)

### Q: Apakah saya harus menguasai matematika tingkat lanjut untuk belajar Data Science?

**A:** Tidak perlu! Kamu cukup memahami konsep dasar seperti mean, variance, dan probability. Tidak perlu menyelam terlalu dalam ke teori matematika. Dengan praktik coding konsisten, kamu akan memahaminya secara natural.

### Q: Data Science vs Data Analysis itu apa bedanya?

**A:**

- **Data Analysis**: Menganalisis data yang sudah ada untuk menemukan insights (jawab: "Apa yang terjadi?")
- **Data Science**: Membangun model untuk memprediksi/mengklasifikasi sesuatu di masa depan (jawab: "Apa yang akan terjadi?")

Analogi: Data Analyst adalah detektif yang menyelidiki kasus, Data Scientist adalah yang membuat sistem prediksi untuk mencegah kasus sebelum terjadi.

### Q: Apakah Python adalah satu-satunya bahasa untuk Data Science?

**A:** Tidak, tapi Python paling populer karena:

- Sintaksis mudah dipelajari
- Library lengkap (Pandas, NumPy, Scikit-learn)
- Komunitas besar dan banyak tutorial

Bahasa lain seperti R dan SQL juga digunakan, tapi Python adalah pilihan terbaik untuk pemula.

### Q: Berapa lama waktu yang dibutuhkan untuk menjadi Data Scientist?

**A:** Tergantung background dan intensitas belajar:

- **Fokus penuh**: 6-12 bulan untuk competency dasar
- **Part-time**: 1-2 tahun
- Terus belajar: Data Science adalah field yang berkembang cepat, selalu ada tools dan teknik baru

### Q: Apakah saya perlu membeli software/tools mahal?

**A:** Sama sekali tidak! Semua yang kamu butuhkan gratis:

- **Google Colab** - Jupyter Notebook gratis dengan GPU
- **Python** - Open source dan gratis
- **Libraries** - Pandas, NumPy, Scikit-learn semuanya gratis
- **Dataset** - Kaggle, UCI ML Repository punya ratusan dataset gratis

### Q: Apa saja pekerjaan yang bisa didapat setelah belajar Data Science?

**A:** Banyak! Beberapa contoh:

- Data Scientist
- Machine Learning Engineer
- Data Analyst
- Analytics Engineer
- Business Intelligence Analyst
- ML Operations (MLOps) Engineer

Perusahaan dari startup hingga tech giant seperti Google, Facebook, Amazon semuanya mencari data professionals.

### Q: Apakah CRISP-DM yang satu-satunya metodologi?

**A:** Tidak, ada metodologi lain seperti:

- **KDD (Knowledge Discovery in Databases)** - Mirip CRISP-DM
- **SEMMA (SAS)** - Dari SAS Institute
- **Agile Analytics** - Pendekatan iteratif modern

Tapi CRISP-DM adalah yang paling umum dan bisa diadaptasi untuk berbagai jenis project.

---

:::note[Catatan]
Materi ini adalah fondasi untuk pertemuan-pertemuan selanjutnya. Pastikan kamu memahami konsep CRISP-DM karena akan menjadi panduan dalam setiap project Data Science.
:::
