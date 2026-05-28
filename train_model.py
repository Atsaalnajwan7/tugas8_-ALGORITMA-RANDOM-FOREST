import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import time

def ketik(teks, delay=0.03):
    """Efek ketik satu per satu"""
    import sys
    for huruf in teks:
        sys.stdout.write(huruf)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def tunggu():
    input("\n  [Tekan ENTER untuk lanjut...]\n")

# ─────────────────────────────────────────
print("\n" + "="*50)
ketik("  ❤️  PREDIKSI PENYAKIT JANTUNG")
ketik("  Algoritma: Random Forest")
print("="*50)
tunggu()

# STEP 1
print("\n📂 STEP 1 — Load Dataset")
time.sleep(0.5)
df = pd.read_csv('heart.csv')
ketik(f"  ✅ Dataset berhasil dimuat!")
ketik(f"  📊 Jumlah data  : {df.shape[0]} baris")
ketik(f"  📋 Jumlah fitur : {df.shape[1]-1} kolom (+ 1 label)")
tunggu()

# STEP 2
print("\n🔍 STEP 2 — Cek Data")
time.sleep(0.5)
ketik("  Kolom dataset:")
for col in df.columns.tolist():
    ketik(f"    • {col}")
    time.sleep(0.05)
ketik(f"\n  Missing values: {df.isnull().sum().sum()} (tidak ada masalah)")
tunggu()

# STEP 3
print("\n⚙️  STEP 3 — Preprocessing")
time.sleep(0.5)
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ketik("  ✅ Fitur dan label dipisahkan")
ketik("  ✅ Normalisasi (StandardScaler) selesai")
tunggu()

# STEP 4
print("\n✂️  STEP 4 — Split Data (80% latih / 20% uji)")
time.sleep(0.5)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
ketik(f"  ✅ Data latih : {X_train.shape[0]} baris")
ketik(f"  ✅ Data uji   : {X_test.shape[0]} baris")
tunggu()

# STEP 5
print("\n🌲 STEP 5 — Melatih Model Random Forest...")
time.sleep(0.5)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

for i in range(1, 6):
    ketik(f"  🔄 Membangun pohon keputusan... ({i*20}%)")
    time.sleep(0.4)

model.fit(X_train, y_train)
ketik("  ✅ Model selesai dilatih!")
tunggu()

# STEP 6
print("\n📈 STEP 6 — Evaluasi Model")
time.sleep(0.5)
y_pred = model.predict(X_test)
akurasi = round(accuracy_score(y_test, y_pred) * 100, 2)

ketik(f"  🎯 Akurasi Model : {akurasi}%")
time.sleep(0.3)

report = classification_report(y_test, y_pred, target_names=["Tidak Berisiko", "Berisiko"])
print()
ketik("  Laporan Klasifikasi:")
print("-"*45)
for baris in report.split('\n'):
    ketik("  " + baris, delay=0.01)
    time.sleep(0.05)
print("-"*45)
tunggu()

# STEP 7
print("\n💾 STEP 7 — Menyimpan Model")
time.sleep(0.5)
joblib.dump(model, 'model.pkl')
ketik("  ✅ model.pkl tersimpan")
time.sleep(0.3)
joblib.dump(scaler, 'scaler.pkl')
ketik("  ✅ scaler.pkl tersimpan")

# Langsung tampilkan output model tanpa tunggu ENTER
print("\n" + "="*50)
ketik("  📦 DETAIL MODEL RANDOM FOREST")
print("="*50)
time.sleep(0.3)
ketik(f"  • Jumlah pohon (n_estimators) : {model.n_estimators}")
time.sleep(0.2)
ketik(f"  • Kedalaman maksimum (max_depth) : {model.max_depth}")
time.sleep(0.2)
ketik(f"  • Jumlah fitur yang dipakai : {model.n_features_in_}")
time.sleep(0.2)
ketik(f"  • Jumlah kelas : {model.n_classes_}")
time.sleep(0.2)
ketik(f"  • Kelas yang diprediksi : {list(model.classes_)}")
time.sleep(0.3)

print("\n" + "-"*50)
ketik("  🌲 Feature Importance (Fitur Paling Berpengaruh):")
print("-"*50)
importances = model.feature_importances_
feature_names = X.columns.tolist()
sorted_idx = np.argsort(importances)[::-1]
for i, idx in enumerate(sorted_idx):
    bar = "█" * int(importances[idx] * 100)
    ketik(f"  {i+1:2}. {feature_names[idx]:<28} {importances[idx]:.4f}  {bar}")
    time.sleep(0.1)
print("-"*50)

# SELESAI
print("\n" + "="*50)
ketik("  🎉 PELATIHAN SELESAI!")
ketik(f"  Akurasi akhir: {akurasi}%")
ketik("  Model siap digunakan di aplikasi Flask.")
print("="*50 + "\n")