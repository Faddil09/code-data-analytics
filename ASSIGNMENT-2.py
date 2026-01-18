# ==========================================
# AI-assisted Device for Early Shingles Screening
# MUHAMMAD FADDIL -U2102252
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ----------------------------
# PHASE I: DATA LOADING
# ----------------------------
df_dirty = pd.read_csv('raw_medical_data_difficult.csv')

# ----------------------------
# PHASE II: DATA CLEANING & FEATURE ENGINEERING
# ----------------------------
df = df_dirty.drop_duplicates().copy()
df = df[(df['Age'].between(0,120)) | (df['Age'].isna())]

# Imputation
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

df['Age'] = num_imputer.fit_transform(df[['Age']])
cols_cat = ['Unilateral_Pain', 'Tingling', 'Fever', 'Rash_Visible']
df[cols_cat] = cat_imputer.fit_transform(df[cols_cat])

# Advanced Encoding (Handles 'UNKNOWN', 'N/A', etc.)
df['Unilateral_Pain'] = df['Unilateral_Pain'].str.upper().str.strip()
encoding_map = {'Y': 1, 'YES': 1, 'N': 0, 'NO': 0, 'N/A': 0, 'UNKNOWN': 0, 'NONE': 0}
df['Unilateral_Pain'] = df['Unilateral_Pain'].replace(encoding_map).astype(float)

# Feature Engineering
df['Symptom_Count'] = df[['Tingling','Fever','Rash_Visible']].sum(axis=1)

# Logic for Target
logic = (df['Unilateral_Pain']*3) + (df['Tingling']*2) + ((df['Age']-50)/10)
df['Is_Shingles'] = (logic > 5.5).astype(int)

# ----------------------------
# PHASE III: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------

# 1. Age Distribution vs Shingles Diagnosis
plt.figure(figsize=(8,4))
sns.boxplot(x='Is_Shingles', y='Age', data=df, palette='Set2')
plt.title('EDA: Age Distribution vs Shingles Diagnosis')
plt.show()

# 2. Symptom Count Across Data
plt.figure(figsize=(8,4))
sns.countplot(x='Symptom_Count', data=df, palette='viridis')
plt.title('EDA: Symptom Count Across Dataset')
plt.xlabel('Number of Symptoms')
plt.ylabel('Patient Count')
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10,6))
corr = df[['Age', 'Unilateral_Pain', 'Tingling', 'Fever', 'Rash_Visible', 'Symptom_Count', 'Is_Shingles']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('EDA: Feature Correlation Heatmap')
plt.show()

# ----------------------------
# PHASE IV: K-MEANS CLUSTERING (Method 1)
# ----------------------------
X_cluster = df[['Age', 'Symptom_Count']]

# 4. Elbow Method for Optimal k
inertia = []
K_range = range(1, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,4))
plt.plot(K_range, inertia, marker='o', color='purple')
plt.title('Method 1: Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
plt.show()

# Final Clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)

# Hollow Circle Scatter Plot
plt.figure(figsize=(9,7))
colors = ['blue','green','red']
for i, color in zip(range(3), colors):
    subset = df[df['Cluster'] == i]
    plt.scatter(subset['Age'], subset['Symptom_Count'], edgecolors=color, 
                facecolors='none', s=40, alpha=0.7, label=f'Cluster {i}')
plt.xlabel('Age')
plt.ylabel('Symptom Count')
plt.title('Method 1: Patient Risk Archetypes (Hollow Circles)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ----------------------------
# PHASE V: NEURAL NETWORK (Method 2)
# ----------------------------
X = df[['Age','Unilateral_Pain','Tingling','Fever','Rash_Visible','Symptom_Count','Cluster']]
y = df['Is_Shingles']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, 
                    validation_data=(X_val_scaled, y_val), verbose=1)

# 5. Accuracy Convergence Plot
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Method 2: Accuracy Convergence (Learning Curve)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 6. Final Confusion Matrix
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Method 2: Final Confusion Matrix')
plt.xlabel('Predicted Diagnosis')
plt.ylabel('Actual Diagnosis')
plt.show()

# Evaluation Summary
test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%")