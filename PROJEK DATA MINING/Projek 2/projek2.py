# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set Style for Visualizations
sns.set(style="whitegrid")

##########################################
# TAHAP 1: DATA PREPARATION
##########################################
print("=" * 50)
print("TAHAP 1: DATA PREPARATION")
print("=" * 50)

# Load Dataset
file_path = 'swiggy_cleaned.csv'  # Filepath dataset
data = pd.read_csv(file_path)

# 1.1 Memeriksa Dimensi dan Struktur Dataset
print("\n1.1 MEMERIKSA DATASET")
print(f"Dimensi dataset: {data.shape}")
print("Lima baris pertama dataset:")
print(data.head())
print("\nInformasi dataset:")
data.info()

# 1.2 Mengecek Missing Values dan Tipe Data
print("\n1.2 CEK KUALITAS DATA")
print("Jumlah nilai yang hilang per kolom:")
print(data.isnull().sum())
print("\nTipe data setiap kolom:")
print(data.dtypes)

# 1.3 Identifikasi Data Duplikat
print("\n1.3 CEK DATA DUPLIKAT")
duplicates = data.duplicated().sum()
print(f"Jumlah duplikat dalam dataset: {duplicates}")

# 1.4 Statistik Deskriptif
print("\n1.4 STATISTIK DESKRIPTIF")
print(data.describe(include="all"))

##########################################
# TAHAP 2: DATA PREPROCESSING
##########################################
print("\n" + "=" * 50)
print("TAHAP 2: DATA PREPROCESSING")
print("=" * 50)

# 2.1 Data Cleaning
print("\n2.1 PEMBERSIHAN DATA")
# Menghapus spasi ekstra
columns_to_strip = ['hotel_name', 'food_type', 'location']
for col in columns_to_strip:
    if col in data.columns:
        data[col] = data[col].str.strip()

# Mengonversi kolom ke numerik (dengan error handling)
numeric_columns = ['rating', 'time_minutes', 'offer_above', 'offer_percentage']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Mengatasi 'offer_percentage' dengan nilai tertentu
data['offer_percentage'] = data['offer_percentage'].replace('not_available', np.nan)

print("\nData setelah pembersihan:")
print(data.info())

# 2.2 Handling Missing Values
print("\n2.2 PENANGANAN MISSING VALUES")
print("Missing values sebelum penanganan:")
print(data.isnull().sum())

# Imputasi nilai null
data['time_minutes'].fillna(data['time_minutes'].median(), inplace=True)
data['offer_percentage'].fillna(0, inplace=True)
data['rating'].fillna(data['rating'].median(), inplace=True)
data['offer_above'].fillna(data['offer_above'].median(), inplace=True)

print("\nMissing values setelah penanganan:")
print(data.isnull().sum())

# 2.3 Feature Engineering
print("\n2.3 FEATURE ENGINEERING")
data['effective_offer'] = (data['offer_above'] * data['offer_percentage']) / 100
print("Fitur baru 'effective_offer' ditambahkan.")

# 2.4 Deteksi dan Penanganan Outlier
print("\n2.4 DETEKSI DAN PENANGANAN OUTLIER")
for col in numeric_columns + ['effective_offer']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"{col}: Batas bawah {lower_bound}, batas atas {upper_bound}")

# 2.5 Feature Scaling
print("\n2.5 FEATURE SCALING")
scaler = MinMaxScaler()
data_scaled = data.copy()
data_scaled[numeric_columns + ['effective_offer']] = scaler.fit_transform(data[numeric_columns + ['effective_offer']])

print("\nContoh data setelah scaling (5 baris pertama):")
print(data_scaled[numeric_columns + ['effective_offer']].head())

# 2.6 Feature Encoding
print("\n2.6 FEATURE ENCODING")
categorical_columns = ['food_type', 'location']
data_encoded = pd.get_dummies(data_scaled, columns=categorical_columns, prefix=categorical_columns)

print("\nBeberapa kolom setelah encoding (contoh):")
encoded_columns = data_encoded.columns
print("Kolom awal:", list(encoded_columns[:5]))
print("Kolom akhir:", list(encoded_columns[-5:]))

##########################################
# TAHAP 3: EXPLORATORY DATA ANALYSIS (EDA)
##########################################
print("\n" + "=" * 50)
print("TAHAP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 3.1 Analisis Statistik Deskriptif
print("\n3.1 STATISTIK DESKRIPTIF")
print(data[numeric_columns + ['effective_offer']].describe())

# 3.2 Distribusi Variabel Kategorikal
print("\n3.2 DISTRIBUSI VARIABEL KATEGORIKAL")
for col in categorical_columns:
    print(f"\nDistribusi {col}:")
    print(data[col].value_counts())

# 3.3 Visualisasi Distribusi Numerik
print("\n3.3 VISUALISASI DISTRIBUSI NUMERIK")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns + ['effective_offer'], 1):
    plt.subplot(3, 2, i)
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# 3.4 Boxplot untuk Analisis Outlier
print("\n3.4 ANALISIS OUTLIERS")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns + ['effective_offer'], 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=data, y=col)
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# 3.5 Korelasi Antar Variabel Numerik
print("\n3.5 ANALISIS KORELASI")
correlation_matrix = data[numeric_columns + ['effective_offer']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
