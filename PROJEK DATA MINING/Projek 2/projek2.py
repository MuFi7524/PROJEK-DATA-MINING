import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

##########################################
# TAHAP 1: DATA PREPARATION
##########################################
print("="*50)
print("TAHAP 1: DATA PREPARATION")
print("="*50)

# 1.1 Membaca Dataset
print("\n1.1 MEMBACA DATASET")
df = pd.read_csv('Stores.csv')
print("Dimensi awal dataset:", df.shape)
print("\nLima data pertama:")
print(df.head())

# 1.2 Cek Kualitas Data
print("\n1.2 CEK KUALITAS DATA")
print("\nMissing Values:")
print(df.isnull().sum())
print("\nJumlah data duplikat:", df.duplicated().sum())
print("\nTipe data setiap kolom:")
print(df.dtypes)

##########################################
# TAHAP 2: DATA PREPROCESSING
##########################################
print("\n" + "="*50)
print("TAHAP 2: DATA PREPROCESSING")
print("="*50)

# 2.1 Pembersihan Data
print("\n2.1 PEMBERSIHAN DATA")
# Membersihkan spasi berlebih
df['Property'] = df['Property'].str.strip()
df['Type'] = df['Type'].str.strip()
df['Old/New'] = df['Old/New'].str.strip()

# Membersihkan format Revenue
df['Revenue'] = df['Revenue'].str.replace(',', '').astype(float)

# 2.2 Penanganan Missing Values
print("\n2.2 PENANGANAN MISSING VALUES")
df['Checkout Number'] = df.groupby('Type')['Checkout Number'].transform(
    lambda x: x.fillna(x.median()))

# 2.3 Feature Engineering
print("\n2.3 FEATURE ENGINEERING")
df['Revenue_per_m2'] = df['Revenue'] / df['AreaStore']
df['Revenue_per_checkout'] = df['Revenue'] / df['Checkout Number']

# 2.4 Feature Scaling
print("\n2.4 FEATURE SCALING")
numeric_columns = ['AreaStore', 'Checkout Number', 'Revenue', 
                   'Revenue_per_m2', 'Revenue_per_checkout']
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 2.5 Feature Encoding
print("\n2.5 FEATURE ENCODING")
df_encoded = pd.get_dummies(df_scaled, 
                            columns=['Property', 'Type', 'Old/New'],
                            prefix=['Property', 'Type', 'Status'])

##########################################
# TAHAP 3: EXPLORATORY DATA ANALYSIS (EDA)
##########################################
print("\n" + "="*50)
print("TAHAP 3: EXPLORATORY DATA ANALYSIS")
print("="*50)

# 3.1 Analisis Statistik Deskriptif
print("\n3.1 ANALISIS STATISTIK DESKRIPTIF")
print(df_encoded[numeric_columns].describe())

# 3.2 Analisis Variabel Kategorikal
print("\n3.2 ANALISIS VARIABEL KATEGORIKAL")
categorical_cols = ['Property', 'Type', 'Old/New']
for col in categorical_cols:
    print(f"\nDistribusi {col}:")
    print(df[col].value_counts())

# 3.3 Visualisasi Distribusi Data Numerik
print("\n3.3 VISUALISASI DISTRIBUSI DATA NUMERIK")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns[:3], 1):  # Original numeric columns
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3.4 Analisis Outliers
print("\n3.4 ANALISIS OUTLIERS")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns[:3], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, y=col)
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# 3.5 Analisis Korelasi
print("\n3.5 ANALISIS KORELASI")
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
