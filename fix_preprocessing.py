import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# =============================================
# FIX PREPROCESSING - Handle Imbalanced Data
# =============================================

# Install dulu jika belum ada:
# pip install imbalanced-learn

# 1. Load dataset
df = pd.read_csv('dataset/online_gaming_behavior_dataset.csv')
print("Dataset loaded:", df.shape)

# 2. Hapus kolom yang tidak relevan
# EngagementLevel dibuang karena terlalu related ke GameDifficulty (data leakage!)
df = df.drop(columns=['PlayerID', 'EngagementLevel'])
print("Kolom PlayerID & EngagementLevel dihapus")

# 3. Encode kolom kategorikal
kategorikal = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
le_dict = {}

for col in kategorikal:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"Encoded {col}: {list(le.classes_)}")

# 4. Pisahkan fitur dan target
X = df.drop(columns=['GameDifficulty'])
y = df['GameDifficulty']

print("\nFitur yang dipakai:", X.columns.tolist())
print("Distribusi target awal:")
print(y.value_counts())

# 5. Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 6. Split dulu sebelum SMOTE (penting! SMOTE hanya di train)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\nSebelum SMOTE - Train: {X_train.shape[0]} baris")
print("Distribusi train sebelum SMOTE:")
print(y_train.value_counts())

# 7. Terapkan SMOTE hanya di data train
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\nSetelah SMOTE - Train: {X_train_sm.shape[0]} baris")
print("Distribusi train setelah SMOTE:")
print(pd.Series(y_train_sm).value_counts())

# 8. Simpan semua
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le_dict, 'model/label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_columns.pkl')

pd.DataFrame(X_train_sm, columns=X.columns).to_csv('dataset/X_train.csv', index=False)
X_val.to_csv('dataset/X_val.csv', index=False)
X_test.to_csv('dataset/X_test.csv', index=False)
pd.Series(y_train_sm, name='GameDifficulty').to_csv('dataset/y_train.csv', index=False)
y_val.to_csv('dataset/y_val.csv', index=False)
y_test.to_csv('dataset/y_test.csv', index=False)

print("\n✅ Fix preprocessing selesai!")