import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# =============================================
# RESET PREPROCESSING - Target: EngagementLevel
# =============================================

# 1. Load dataset
df = pd.read_csv('dataset/online_gaming_behavior_dataset.csv')
print("Dataset loaded:", df.shape)

# 2. Hapus kolom tidak relevan
df = df.drop(columns=['PlayerID'])
print("Kolom PlayerID dihapus")

# 3. Encode kolom kategorikal
kategorikal = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
le_dict = {}

for col in kategorikal:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"Encoded {col}: {list(le.classes_)}")

# 4. Pisahkan fitur dan target (TARGET SEKARANG: EngagementLevel)
X = df.drop(columns=['EngagementLevel'])
y = df['EngagementLevel']

print("\nFitur yang dipakai:", X.columns.tolist())
print("\nDistribusi target (EngagementLevel):")
print(y.value_counts())

# 5. Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 6. Split 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\nUkuran data:")
print(f"  Train    : {X_train.shape[0]} baris (70%)")
print(f"  Validasi : {X_val.shape[0]} baris (15%)")
print(f"  Test     : {X_test.shape[0]} baris (15%)")

# 7. Simpan semua
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le_dict, 'model/label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_columns.pkl')

X_train.to_csv('dataset/X_train.csv', index=False)
X_val.to_csv('dataset/X_val.csv', index=False)
X_test.to_csv('dataset/X_test.csv', index=False)
y_train.to_csv('dataset/y_train.csv', index=False)
y_val.to_csv('dataset/y_val.csv', index=False)
y_test.to_csv('dataset/y_test.csv', index=False)

print("\n✅ Reset preprocessing selesai!")