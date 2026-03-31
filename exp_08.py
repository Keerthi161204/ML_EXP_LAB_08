import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

from sklearn.metrics import make_scorer, mean_squared_error

# ---------------------------
# LOAD DATA (substitute dataset)
# ---------------------------
data = load_breast_cancer()
X = data.data
y_class = data.target

# create fake regression target (since dataset doesn't have one)
y_reg = X[:, 0] * 10 + np.random.randn(len(X))

# ---------------------------
# STANDARDIZE
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# PCA (95% variance)
# ---------------------------
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("Original features:", X.shape[1])
print("Reduced features:", X_pca.shape[1])

# ---------------------------
# REGRESSION
# ---------------------------
mse_scorer = make_scorer(mean_squared_error)

# Linear Regression
lr = LinearRegression()

lr_mse_no_pca = cross_val_score(lr, X_scaled, y_reg, cv=5, scoring=mse_scorer).mean()
lr_mse_pca = cross_val_score(lr, X_pca, y_reg, cv=5, scoring=mse_scorer).mean()

lr_r2_no_pca = cross_val_score(lr, X_scaled, y_reg, cv=5, scoring='r2').mean()
lr_r2_pca = cross_val_score(lr, X_pca, y_reg, cv=5, scoring='r2').mean()

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=5)

rf_mse_no_pca = cross_val_score(rf, X_scaled, y_reg, cv=5, scoring=mse_scorer).mean()
rf_mse_pca = cross_val_score(rf, X_pca, y_reg, cv=5, scoring=mse_scorer).mean()

rf_r2_no_pca = cross_val_score(rf, X_scaled, y_reg, cv=5, scoring='r2').mean()
rf_r2_pca = cross_val_score(rf, X_pca, y_reg, cv=5, scoring='r2').mean()

# ---------------------------
# CLASSIFICATION
# ---------------------------
# Logistic Regression
log = LogisticRegression(max_iter=1000)

log_acc_no_pca = cross_val_score(log, X_scaled, y_class, cv=5, scoring='accuracy').mean()
log_acc_pca = cross_val_score(log, X_pca, y_class, cv=5, scoring='accuracy').mean()

log_f1_no_pca = cross_val_score(log, X_scaled, y_class, cv=5, scoring='f1').mean()
log_f1_pca = cross_val_score(log, X_pca, y_class, cv=5, scoring='f1').mean()

# SVM
svm = SVC(kernel='rbf')

svm_acc_no_pca = cross_val_score(svm, X_scaled, y_class, cv=5, scoring='accuracy').mean()
svm_acc_pca = cross_val_score(svm, X_pca, y_class, cv=5, scoring='accuracy').mean()

svm_f1_no_pca = cross_val_score(svm, X_scaled, y_class, cv=5, scoring='f1').mean()
svm_f1_pca = cross_val_score(svm, X_pca, y_class, cv=5, scoring='f1').mean()

# ---------------------------
# PRINT RESULTS
# ---------------------------
print("\n--- REGRESSION ---")
print("LR MSE:", lr_mse_no_pca, "->", lr_mse_pca)
print("LR R2:", lr_r2_no_pca, "->", lr_r2_pca)

print("RF MSE:", rf_mse_no_pca, "->", rf_mse_pca)
print("RF R2:", rf_r2_no_pca, "->", rf_r2_pca)

print("\n--- CLASSIFICATION ---")
print("Logistic Acc:", log_acc_no_pca, "->", log_acc_pca)
print("Logistic F1:", log_f1_no_pca, "->", log_f1_pca)

print("SVM Acc:", svm_acc_no_pca, "->", svm_acc_pca)
print("SVM F1:", svm_f1_no_pca, "->", svm_f1_pca)
