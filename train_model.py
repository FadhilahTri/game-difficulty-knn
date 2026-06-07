import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib

# =============================================
# STEP 4 - TRAINING KNN & TUNING NILAI K
# =============================================

# 1. Load data hasil preprocessing
X_train = pd.read_csv('dataset/X_train.csv')
X_val   = pd.read_csv('dataset/X_val.csv')
X_test  = pd.read_csv('dataset/X_test.csv')
y_train = pd.read_csv('dataset/y_train.csv').squeeze()
y_val   = pd.read_csv('dataset/y_val.csv').squeeze()
y_test  = pd.read_csv('dataset/y_test.csv').squeeze()

print("✅ Data berhasil di-load!")
print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# =============================================
# 2. ELBOW METHOD - Cari nilai K optimal
# =============================================
print("\n🔍 Mencari nilai K optimal (Elbow Method)...")

k_range = range(1, 21)
val_scores = []
train_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
    val_scores.append(accuracy_score(y_val, knn.predict(X_val)))
    print(f"   K={k:2d} → Train: {train_scores[-1]:.4f} | Val: {val_scores[-1]:.4f}")

# Plot Elbow
plt.figure(figsize=(10, 5))
plt.plot(k_range, train_scores, 'b-o', label='Train Accuracy')
plt.plot(k_range, val_scores, 'r-o', label='Validation Accuracy')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi')
plt.title('Elbow Method - Mencari K Optimal')
plt.legend()
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.savefig('dataset/plot_elbow.png')
plt.show()

# Pilih K terbaik berdasarkan validasi
best_k = k_range[np.argmax(val_scores)]
print(f"\n🏆 K optimal: {best_k} (Val Accuracy: {max(val_scores):.4f})")

# =============================================
# 3. TRAINING MODEL FINAL dengan K terbaik
# =============================================
print(f"\n🚀 Training model final dengan K={best_k}...")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)

# Evaluasi di data test
y_pred = knn_final.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n📊 HASIL EVALUASI MODEL FINAL:")
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =============================================
# 4. CONFUSION MATRIX
# =============================================
le = joblib.load('model/label_encoders.pkl')
class_names = le['EngagementLevel'].classes_

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title(f'Confusion Matrix (K={best_k})')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.tight_layout()
plt.savefig('dataset/plot_confusion_matrix.png')
plt.show()

# =============================================
# 5. SIMPAN MODEL
# =============================================
joblib.dump(knn_final, 'model/knn_model.pkl')
joblib.dump(best_k, 'model/best_k.pkl')

print(f"\n✅ Model tersimpan di model/knn_model.pkl")
print(f"✅ Best K={best_k} tersimpan di model/best_k.pkl")
print("\n🎉 Step 4 selesai!")