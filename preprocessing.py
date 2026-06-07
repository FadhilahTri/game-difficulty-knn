import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# =============================================
# STEP 3 - PREPROCESSING DATA
# =============================================

# 1. Load dataset
df = pd.read_csv('dataset/online_gaming_behavior_dataset.csv')
print("Dataset loaded:", df.shape)

# 2. Hapus kolom yang tidak dipakai
df = df.drop(columns=['PlayerID'])
print("Kolom PlayerID dihapus")

# 3. Pisahkan fitur kategorikal & numerik
print("\nKolom sebelum encoding:")
print(df.dtypes)

# 4. Encode kolom kategorikal (selain target)
kategorikal = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
le_dict = {}

for col in kategorikal:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"Encoded {col}: {list(le.classes_)}")

# 5. Pisahkan fitur (X) dan target (y)
X = df.drop(columns=['GameDifficulty'])
y = df['GameDifficulty']

print("\nFitur yang dipakai:")
print(X.columns.tolist())
print("\nDistribusi target setelah encoding:")
print(y.value_counts())

# 6. Normalisasi fitur numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("\nNormalisasi selesai!")
print(X_scaled.describe().round(2))

# 7. Split data: 70% train, 15% validasi, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\nUkuran data:")
print(f"  Train      : {X_train.shape[0]} baris ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  Validasi   : {X_val.shape[0]} baris ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"  Test       : {X_test.shape[0]} baris ({X_test.shape[0]/len(df)*100:.1f}%)")

# 8. Simpan scaler dan data hasil preprocessing
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le_dict, 'model/label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_columns.pkl')

# Simpan data split ke file
X_train.to_csv('dataset/X_train.csv', index=False)
X_val.to_csv('dataset/X_val.csv', index=False)
X_test.to_csv('dataset/X_test.csv', index=False)
y_train.to_csv('dataset/y_train.csv', index=False)
y_val.to_csv('dataset/y_val.csv', index=False)
y_test.to_csv('dataset/y_test.csv', index=False)

print("\n✅ Preprocessing selesai!")
print("✅ scaler.pkl tersimpan di folder model/")
print("✅ Data train/val/test tersimpan di folder dataset/")