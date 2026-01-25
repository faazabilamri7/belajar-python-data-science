---
title: Classification Metrics & Evaluation
description: Metrik evaluasi untuk classification problems
sidebar:
  order: 3
---

## 🎯 Classification Metrics Overview

Untuk classification problems, kita punya berbagai metrics untuk evaluate model. Memilih metric yang tepat sangat penting untuk understand seberapa baik model bekerja.

### Confusion Matrix Foundation

Semua classification metrics dibangun dari **Confusion Matrix**:

```
                    PREDICTED
                 Positive   Negative
ACTUAL Positive      TP        FN
       Negative      FP        TN
```

| Istilah | Meaning |
| ------- | ------- |
| **TP** | True Positive - Correctly predicted positive |
| **TN** | True Negative - Correctly predicted negative |
| **FP** | False Positive - Incorrectly predicted positive |
| **FN** | False Negative - Incorrectly predicted negative |

---

## 📊 Confusion Matrix in Practice

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Hitung confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Output:
# [[85  5]      TN=85, FP=5
#  [15 95]]     FN=15, TP=95

# Visualisasi
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

---

## 📈 Accuracy

**Definition:** Persentase prediksi yang benar dari total prediksi.

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # Output: 0.8500

# Interpretation: 85% dari semua prediksi kita benar
```

### Kapan Menggunakan Accuracy?

- ✅ Dataset balanced (class distribution sama)
- ❌ Dataset imbalanced (misleading result)

### Example: Balanced vs Imbalanced

```
BALANCED DATASET (50% positive, 50% negative):
- Accuracy = 85% ← Good metric

IMBALANCED DATASET (95% negative, 5% positive):
- Model always predict negative → Accuracy = 95% ❌
- Tapi tidak mendeteksi positives sama sekali!
```

---

## 🎯 Precision

**Definition:** Dari semua prediksi positif, berapa persen yang benar-benar positif?

$$Precision = \frac{TP}{TP + FP}$$

"Ketika model memprediksi POSITIVE, berapa reliable?"

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

# Interpretation:
# Dari 100 prediksi positif, 90 benar positif (TP), 10 salah positif (FP)
# Precision = 90 / (90 + 10) = 0.90
```

### Kapan Menggunakan Precision?

- ✅ False Positive costly (spam detection, credit approval)
- Contoh: Email spam filter
  - FP (mark legitimate as spam) = BAD ❌
  - Lebih baik miss beberapa spam daripada block email penting

---

## 🔍 Recall (Sensitivity)

**Definition:** Dari semua actual positives, berapa persen yang berhasil kita prediksi?

$$Recall = \frac{TP}{TP + FN}$$

"Model berhasil catch berapa persen dari semua positive cases?"

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}")

# Interpretation:
# Ada 100 actual positives
# Model correctly identified 85 of them
# Recall = 85 / 100 = 0.85
```

### Kapan Menggunakan Recall?

- ✅ False Negative costly (disease detection, fraud detection)
- Contoh: Medical diagnosis
  - FN (miss disease) = VERY BAD ❌
  - Lebih baik false alarm daripada miss penyakit

---

## ⚖️ Precision vs Recall Trade-off

```
HIGH PRECISION, LOW RECALL:
- Conservative predictions
- Only predict positive jika sangat confident
- Few false alarms, tapi banyak missed positives

BALANCED:
- Not too conservative, not too aggressive
- Good for most cases

LOW PRECISION, HIGH RECALL:
- Aggressive predictions
- Predict positive lebih sering
- Catch most positives, tapi banyak false alarms
```

---

## 🎯 F1-Score

**Definition:** Harmonic mean dari Precision dan Recall.

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

F1-Score adalah metrik yang balance antara precision dan recall. Berguna untuk imbalanced dataset atau ketika kita butuh balance antara dua metrics.

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

# Range: 0 to 1
# 1 = perfect, 0 = worst
```

### Kapan Menggunakan F1?

- ✅ Imbalanced dataset
- ✅ Butuh balance antara precision & recall
- ✅ FP dan FN sama penting

### Comparison Table

```
              Precision  Recall  F1
Model A         0.90     0.70   0.79
Model B         0.80     0.80   0.80
Model C         0.70     0.90   0.79

Best for balanced? Model B (F1=0.80)
```

---

## 📋 Classification Report

**Lengkap:** Semua metrics dalam satu report

```python
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, 
                              target_names=['Negative', 'Positive'])
print(report)

# Output:
#               precision    recall  f1-score   support
#
#     Negative       0.85      0.94      0.89       500
#     Positive       0.90      0.76      0.83       200
#
#    accuracy                           0.87       700
#   macro avg       0.88      0.85      0.86       700
# weighted avg       0.87      0.87      0.87       700
```

**Interpretasi:**
- **Precision (Negative):** 85% dari negative predictions benar
- **Recall (Positive):** 76% dari actual positives terdeteksi
- **Support:** Jumlah samples di setiap class
- **Weighted avg:** Average weighted by support

---

## 📊 ROC Curve & AUC

**ROC (Receiver Operating Characteristic):** Plot True Positive Rate vs False Positive Rate

$$TPR = \frac{TP}{TP + FN}$$ (sama dengan Recall)

$$FPR = \frac{FP}{FP + TN}$$

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Model harus output probability (tidak semua bisa)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"AUC Score: {auc_score:.4f}")
```

### Interpretasi AUC

| AUC | Interpretasi |
| --- | ------------ |
| 0.5 | Random (no discrimination) |
| 0.6-0.7 | Fair |
| 0.7-0.8 | Good |
| 0.8-0.9 | Excellent |
| 0.9-1.0 | Outstanding |

---

## 🎪 Choosing the Right Metric

### Decision Tree

```
Problem Type?
│
├─ Balanced Classification?
│  ├─ Yes → Use Accuracy
│  └─ No → Use F1-Score or ROC-AUC
│
├─ False Positive Costly?
│  └─ Yes → Use Precision
│
├─ False Negative Costly?
│  └─ Yes → Use Recall
│
└─ Need to Compare Models?
   └─ Use ROC-AUC (threshold-independent)
```

### Real-World Examples

```
SPAM DETECTION:
- False Positive (block email) = BAD
- → Use Precision
- "90% confidence when marking as spam"

DISEASE SCREENING:
- False Negative (miss disease) = BAD
- → Use Recall
- "Catch 95% of actual diseases"

CREDIT APPROVAL:
- Balanced concern
- → Use F1-Score or ROC-AUC
```

---

## 🔍 Handling Imbalanced Data

### Problem

```python
# Imbalanced dataset
print(y.value_counts())
# Negative    950  (95%)
# Positive     50  (5%)

# Model always predict Negative
# Accuracy = 95% ❌ (misleading!)
# But catches 0% of actual positives
```

### Solutions

#### 1. Stratified Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y  # ← Maintain ratio
)
```

#### 2. Different Metric

```python
# Instead of accuracy
accuracy = accuracy_score(y_test, y_pred)  # ❌ Misleading

# Use F1 or ROC-AUC
f1 = f1_score(y_test, y_pred)  # ✅ Better
auc = roc_auc_score(y_test, y_pred_proba)  # ✅ Best
```

#### 3. Class Weights

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Automatically adjust for imbalance
```

#### 4. Resampling

```python
from imblearn.over_sampling import SMOTE

# Oversample minority class
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model.fit(X_resampled, y_resampled)
```

---

## 📊 Complete Evaluation Example

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate all metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Print summary
print("="*50)
print("COMPREHENSIVE EVALUATION")
print("="*50)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## ✏️ Latihan

### Latihan 1: Calculate Metrics

```python
# Given confusion matrix:
# TP=80, TN=85, FP=15, FN=20

# Calculate:
# 1. Accuracy
# 2. Precision
# 3. Recall
# 4. F1-Score
```

### Latihan 2: Which Metric?

Untuk setiap scenario, pilih metric terbaik dan jelaskan why:

1. Fraud detection
2. Medical diagnosis
3. Product recommendation
4. Credit approval

### Latihan 3: Compare Models

```python
# Train 3 models on Titanic dataset
# Evaluate dengan semua metrics
# Create comparison table
# Recommend best model dengan justifikasi
```

---

## 🔗 Referensi

- [Scikit-Learn Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Understanding Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a-simple-way-52461ba0135f)
- [ROC Curves Explained](https://towardsdatascience.com/understanding-roc-curves-2ad4e75feaea)
