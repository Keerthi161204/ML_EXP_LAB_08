# ML_EXP_LAB_08

# 📘 PCA for Dimensionality Reduction (Regression & Classification)

## 🔍 Overview

This project explores the impact of **Principal Component Analysis (PCA)** on both:

* **Regression tasks**
* **Classification tasks**

The goal is to evaluate how dimensionality reduction affects model performance in terms of accuracy, error, and generalization.

---

## 🎯 Objectives

* Apply **Standardization** and **PCA** (95% variance retention)
* Compare model performance **before and after PCA**
* Evaluate using:

  * Regression: MSE, R²
  * Classification: Accuracy, F1-score

---

## 📂 Dataset

* Dataset: **Breast Cancer Wisconsin Dataset** (Scikit-learn)
* Samples: 569
* Features: 30 numerical features

### ⚠️ Important Note

* The dataset is originally for **classification only**
* A **synthetic regression target** is created:

  ```
  y_reg = X[:, 0] * 10 + noise
  ```

---

## ⚙️ Implementation Details

### 🔹 Preprocessing

* Standardization using `StandardScaler`
* PCA applied to retain **95% variance**

```
Original features: 30  
Reduced features: ~10–15 (depends on variance)
```

---

## 🧠 Models Used

### 🔸 Regression Models

* Linear Regression
* Random Forest Regressor

### 🔸 Classification Models

* Logistic Regression
* Support Vector Machine (RBF kernel)

---

## 📊 Evaluation Metrics

### Regression

* Mean Squared Error (MSE)
* R² Score

### Classification

* Accuracy
* F1-score

---

## 📈 Results (Typical Trends)

### 🔹 Regression

* Linear Regression:

  * Slight drop or similar performance after PCA
* Random Forest:

  * Often **worse with PCA** (trees don’t benefit from dimensionality reduction)

---

### 🔹 Classification

* Logistic Regression:

  * Often improves or remains stable after PCA
* SVM:

  * Can improve due to reduced noise and dimensionality

---

## 📌 Key Observations

* PCA reduces feature space while preserving most variance

* Works well with:

  * Linear models (Logistic Regression)
  * Distance-based models (SVM)

* Does **not help tree-based models** like Random Forest

* Trade-off:

  * Less computation
  * Possible information loss

---

## 🧠 Conclusion

PCA is useful for reducing dimensionality and improving efficiency, especially for linear and distance-based models. However, it may degrade performance for tree-based models, which rely on original feature structure.

---

## 🚀 How to Run

```bash id="pca23k"
pip install numpy pandas scikit-learn
python your_script_name.py
```

