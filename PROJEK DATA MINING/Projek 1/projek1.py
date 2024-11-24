import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Data Preparation ---
# Membaca dataset
df = pd.read_csv("file_dataset.csv")

# Menampilkan informasi awal
print("20 Data Pertama:")
print(df.head(20))

print("\nDimensi data:")
print(df.shape)

print("\nTipe data setiap atribut:")
print(df.dtypes)

print("\nDeskripsi statistik data:")
print(df.describe())

# Periksa nilai yang hilang
print("\nJumlah nilai yang hilang di setiap kolom:")
print(df.isnull().sum())

# Periksa duplikasi data
print("\nJumlah duplikasi data:")
print(df.duplicated().sum())

# --- Data Preprocessing ---
# Menangani nilai nol (asumsikan kolom-kolom tertentu)
columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_fix:
    mean_value = df[col].mean()
    df[col] = df[col].replace(0, mean_value)

print("\nData setelah mengganti nilai 0 dengan rata-rata pada kolom tertentu:")
print(df.head())

# Periksa outlier menggunakan IQR
print("\nDeteksi Outlier (berdasarkan IQR):")
for col in columns_to_fix:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    print(f"Kolom {col}: {outliers} outlier ditemukan.")

# Rescaling data
scaler = MinMaxScaler()
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nData setelah rescale menggunakan MinMaxScaler:")
print(df_scaled.head())

# --- Exploratory Data Analysis (EDA) ---
# Distribusi kelas pada kolom 'Outcome'
if 'Outcome' in df.columns:
    print("\nDistribusi kelas pada kolom 'Outcome':")
    print(df['Outcome'].value_counts())
    sns.countplot(x='Outcome', data=df)
    plt.title("Distribusi Kelas Outcome")
    plt.show()
else:
    print("\nKolom 'Outcome' tidak ditemukan dalam dataset.")

# Visualisasi distribusi data
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribusi Kolom {col}")
    plt.show()

# Korelasi dan heatmap
print("\nMatriks Korelasi:")
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()

# Scatter plot contoh (hubungan antara BMI dan Glucose)
sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=df)
plt.title("Hubungan BMI dan Glucose")
plt.show()
