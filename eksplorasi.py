import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================
# STEP 2 - EKSPLORASI DATA (EDA)
# =============================================

# 1. Load dataset
df = pd.read_csv('dataset/online_gaming_behavior_dataset.csv')

# 2. Info dasar dataset
print("=" * 50)
print("SHAPE DATASET (baris x kolom):")
print(df.shape)

print("\nNAMA KOLOM:")
print(df.columns.tolist())

print("\n5 DATA PERTAMA:")
print(df.head())

print("\nINFO TIPE DATA:")
print(df.info())

print("\nCEK MISSING VALUES:")
print(df.isnull().sum())

print("\nSTATISTIK DESKRIPTIF:")
print(df.describe())

# 3. Distribusi kolom target
print("\nDISTRIBUSI KELAS TARGET (GameDifficulty):")
print(df['GameDifficulty'].value_counts())

# 4. Visualisasi distribusi kelas
plt.figure(figsize=(6, 4))
sns.countplot(x='GameDifficulty', data=df, palette='Set2')
plt.title('Distribusi Tingkat Kesulitan Game')
plt.xlabel('Game Difficulty')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.savefig('dataset/plot_distribusi_kelas.png')
plt.show()

# 5. Heatmap korelasi fitur numerik
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Heatmap Korelasi Fitur Numerik')
plt.tight_layout()
plt.savefig('dataset/plot_korelasi.png')
plt.show()

print("\nDone! Plot tersimpan di folder dataset/")