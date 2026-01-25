---
title: Tools & Teknologi Data Science
description: Tools, bahasa pemrograman, dan environment yang digunakan
sidebar:
  order: 4
---

## 🛠️ Tools & Teknologi Data Science

Untuk sukses di Data Science, kamu perlu familiar dengan berbagai tools dan teknologi. Tapi jangan khawatir - ini gradual! Mulai dari yang simple, lalu berkembang.

---

## 🐍 Bahasa Pemrograman

### Python (Rekomendasi Utama)

| Aspek | Rating | Catatan |
| ----- | ------ | ------- |
| Kemudahan Belajar | ⭐⭐⭐⭐⭐ | Syntax bersih dan intuitif |
| Library | ⭐⭐⭐⭐⭐ | Pandas, NumPy, Scikit-learn paling lengkap |
| Community | ⭐⭐⭐⭐⭐ | Komunitas besar, banyak tutorial |
| Industry Adoption | ⭐⭐⭐⭐⭐ | De facto standard untuk DS/ML |

**Mengapa Python?**
- 📚 Library ekosistem yang matang
- 🚀 Cepat untuk prototyping
- 🤝 Komunitas aktif dan besar
- 💼 Digunakan di perusahaan-perusahaan top

### R (untuk Statistical Analysis)

| Aspek | Rating | Catatan |
| ----- | ------ | ------- |
| Kemudahan Belajar | ⭐⭐⭐ | Syntax agak berbeda |
| Library | ⭐⭐⭐⭐⭐ | Excellent untuk stats |
| Community | ⭐⭐⭐⭐ | Komunitas akademik kuat |
| Industry Adoption | ⭐⭐⭐ | Less common than Python |

**Kapan gunakan R?**
- Fokus pada statistical analysis
- Butuh advanced visualization
- Working dengan academic teams

### SQL (Essential!)

Meskipun bukan "bahasa pemrograman", SQL **wajib** dikuasai:

```sql
-- Example query
SELECT customer_id, COUNT(*) as purchase_count
FROM transactions
WHERE DATE(purchase_date) >= '2024-01-01'
GROUP BY customer_id
HAVING COUNT(*) > 10
ORDER BY purchase_count DESC;
```

**Kenapa SQL?**
- Database adalah sumber data
- Setiap data scientist perlu query data sendiri
- Fast dan efficient untuk large datasets

---

## 📦 Python Libraries

### Data Manipulation & Analysis

**Pandas**
```python
import pandas as pd

# Read data
df = pd.read_csv('data.csv')

# Basic operations
df['new_col'] = df['col1'] + df['col2']
df_filtered = df[df['age'] > 25]
grouped = df.groupby('category')['sales'].sum()
```

**NumPy**
```python
import numpy as np

# Array operations
arr = np.array([1, 2, 3, 4, 5])
matrix = np.random.randn(3, 4)  # Random 3x4 matrix
result = np.dot(matrix, vector)  # Matrix multiplication
```

### Visualization

**Matplotlib** - Dasar, fleksibel
```python
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.title('My Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
```

**Seaborn** - High-level, cantik
```python
import seaborn as sns

sns.scatterplot(data=df, x='age', y='salary', hue='department')
sns.heatmap(correlation_matrix, annot=True)
```

**Plotly** - Interactive, web-ready
```python
import plotly.express as px

fig = px.scatter(df, x='age', y='salary', hover_data=['name'])
fig.show()
```

### Machine Learning

**Scikit-learn** - Gold standard untuk ML di Python
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

**TensorFlow / Keras** - Deep Learning
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
```

### Statistical Analysis

**SciPy**
```python
from scipy import stats

# Hypothesis testing
t_stat, p_value = stats.ttest_ind(group1, group2)

# Distribution fitting
params = stats.norm.fit(data)
```

**Statsmodels**
```python
import statsmodels.api as sm

# Linear regression dengan statistics
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())  # Detailed statistical summary
```

---

## 💻 Development Environments

### Google Colab (Rekomendasi untuk Pemula!)

**Kelebihan:**
- ✅ Gratis dan tidak perlu install
- ✅ GPU/TPU gratis
- ✅ Terintegrasi dengan Google Drive
- ✅ Semua library sudah terinstall

**Kekurangan:**
- ❌ Hanya cocok untuk notebook-style coding
- ❌ Tidak ideal untuk production

**Cara Pakai:**
1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Login dengan Google account
3. Create new notebook
4. Mulai coding!

### Jupyter Notebook

**Kelebihan:**
- ✅ Interactive notebook
- ✅ Bisa mix code dan markdown
- ✅ Great untuk exploration dan teaching
- ✅ Export ke berbagai format

**Instalasi:**
```bash
pip install jupyter
jupyter notebook
```

### VS Code

**Kelebihan:**
- ✅ Full-featured code editor
- ✅ Git integration
- ✅ Extensions ecosystem
- ✅ Scalable untuk production code

**Extensions penting:**
- Python
- Pylance
- Jupyter
- Git History

### PyCharm

**Kelebihan:**
- ✅ IDE khusus Python yang powerful
- ✅ Built-in debugging
- ✅ Git integration
- ✅ Virtual environment management

**Kekurangan:**
- ❌ Heavier than VS Code
- ❌ Community edition limited features

---

## 📊 Contoh Setup Minimal

Untuk mulai dengan Data Science, kamu hanya perlu:

### Option 1: Google Colab (Paling Mudah)
```
1. Buka colab.research.google.com
2. Langsung bisa coding - no setup needed!
```

### Option 2: Local Setup
```bash
# Create virtual environment
python -m venv ds_env
source ds_env/bin/activate  # Linux/Mac
# atau
ds_env\Scripts\activate  # Windows

# Install essential packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Start Jupyter
jupyter notebook
```

---

## 🔄 Version Control dengan Git

**Kenapa Git?**
- Track changes di code
- Collaborate dengan team
- Revert ke versi sebelumnya jika ada error

**Basic Commands:**
```bash
# Initialize repo
git init

# Add files
git add .

# Commit
git commit -m "Initial commit"

# Push to remote
git push origin main

# Pull latest changes
git pull origin main
```

**Platforms:**
- GitHub - Most popular
- GitLab - Good alternative
- Bitbucket - Atlassian ecosystem

---

## 📚 Learning Path

### Fase 1: Foundations (Minggu 1-2)
- ✅ Google Colab untuk eksperimen
- ✅ Python basics (variables, loops, functions)
- ✅ NumPy untuk array operations

### Fase 2: Data Manipulation (Minggu 3-4)
- ✅ Pandas untuk data analysis
- ✅ SQL untuk database queries
- ✅ Basic visualization dengan Matplotlib

### Fase 3: Analysis & Visualization (Minggu 5-6)
- ✅ EDA (Exploratory Data Analysis)
- ✅ Advanced visualization dengan Seaborn
- ✅ Statistical analysis

### Fase 4: Machine Learning (Minggu 7-10)
- ✅ Scikit-learn basics
- ✅ Model training dan evaluation
- ✅ Hyperparameter tuning

### Fase 5: Advanced (Minggu 11+)
- ✅ Deep Learning dengan TensorFlow
- ✅ Model deployment
- ✅ Production considerations

---

## 📝 Ringkasan Tools Wajib Dikuasai

| Tool | Priority | Alasan |
| ---- | -------- | ------ |
| Python | 🔴 Critical | Core language untuk DS |
| SQL | 🔴 Critical | Access data in databases |
| Pandas | 🔴 Critical | Data manipulation |
| Matplotlib/Seaborn | 🟡 Important | Visualization |
| Scikit-learn | 🟡 Important | Machine Learning |
| Jupyter | 🟡 Important | Interactive notebook |
| Git | 🟡 Important | Version control |
| TensorFlow | 🟢 Optional | Deep Learning (later) |

---

## ✏️ Latihan

### Latihan 1: Setup Environment

1. Pilih salah satu environment (Google Colab atau local)
2. Install necessary libraries
3. Verify dengan test code:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Quick test
df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
print("✅ Setup successful!")
print(df.head())
```

### Latihan 2: Simple Analysis

```python
# Load iris dataset
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Tasks:
# 1. Hitung mean setiap kolom
# 2. Find maximum value
# 3. Find minimum value
# 4. Basic visualization dengan plt.scatter
```

---

## 🔗 Referensi

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)
- [Git & GitHub Guide](https://guides.github.com/)
