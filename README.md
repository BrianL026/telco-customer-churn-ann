# Telco Customer Churn Prediction - ANN Model UI

Aplikasi desktop berbasis Graphical User Interface (GUI) untuk memvisualisasikan, melatih, dan menggunakan model Artificial Neural Network (ANN) MLPClassifier untuk memprediksi churn pelanggan (kemungkinan pelanggan berhenti menggunakan layanan) berdasarkan dataset Telco Customer Churn.

## 🛠️ Struktur File Proyek

Berikut adalah struktur folder proyek yang rapi dan modular:

```text
ANN/
├── .gitignore
├── README.md
├── gui_app.spec
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset pelanggan
├── notebooks/
│   ├── churn_ann.ipynb                        # Eksperimen model (Jupyter Notebook)
│   └── churn_ann.py                           # Eksperimen model (Python Script)
└── src/
    ├── __init__.py
    ├── logic/                                 # Modul logika bisnis & model ANN
    │   ├── __init__.py
    │   ├── model_handler.py                   # Handler data & model
    │   ├── churn_model.joblib                 # Bobot model ANN terlatih
    │   ├── scaler.joblib                      # Scaler penormalan fitur
    │   └── model_features.joblib              # Daftar kolom input model
    └── ui/                                    # Modul antarmuka pengguna (GUI)
        ├── __init__.py
        └── gui_app.py                         # Aplikasi GUI utama
```

### Rincian Komponen Utama:
1. **`src/logic/model_handler.py`**
   - Mengelola pemuatan data, pra-pemrosesan data mentah, pelatihan model ANN, evaluasi (akurasi, precision, recall, confusion matrix), serta penyimpanan dan pemuatan model ke disk.
   - Mengubah input data mentah dari form UI menjadi format data yang siap diprediksi oleh model dengan standardisasi scaler yang sama.

2. **`src/ui/gui_app.py`**
   - Antarmuka utama aplikasi menggunakan pustaka **CustomTkinter** yang modern dan responsif.
   - Fitur utama:
     - **Dashboard & Model**: Menampilkan arsitektur ANN, performa metrik (Akurasi, F1-Score, dll.), tombol latih ulang model, serta plot interaktif kurva Loss dan Confusion Matrix menggunakan Matplotlib.
     - **Predict Churn**: Form input lengkap dengan drop-down, slider, dan perhitungan otomatis biaya total (`TotalCharges`). Menghasilkan kartu status risiko churn (merah untuk risiko tinggi, hijau untuk risiko rendah) beserta analisis faktor risiko.
     - **Dataset Explorer**: Menampilkan ringkasan statistik dataset dan tabel preview data interaktif (50 baris pertama).
     - **Theme Toggle**: Pilihan mode tampilan (Dark Mode / Light Mode / System Mode) yang langsung merubah tema aplikasi secara real-time.
     - **Asynchronous Processing**: Proses training model berjalan di background thread sehingga UI tidak membeku (freeze) saat melatih model.

---

## 🚀 Cara Menjalankan Aplikasi

1. **Aktifkan Virtual Environment (jika belum)**
   Buka terminal/PowerShell di direktori proyek dan jalankan:
   ```powershell
   .venv\Scripts\activate
   ```

2. **Jalankan Aplikasi GUI**
   Jalankan file `gui_app.py` menggunakan Python dari folder root:
   ```bash
   python src/ui/gui_app.py
   ```

---

## 📊 Detail Input Prediksi Churn

Form prediksi menyediakan masukan untuk semua 19 fitur pelanggan:
* **Demografi**: Jenis Kelamin, Lansia (Senior Citizen), Status Pasangan (Partner), Status Tanggungan (Dependents).
* **Layanan Langganan**: Telepon, Saluran Ganda, Layanan Internet (Fiber optic/DSL/No), Keamanan Online, Backup Online, Proteksi Perangkat, Prioritas Dukungan Teknis, TV Streaming, Film Streaming.
* **Kontrak & Biaya**: Durasi Kontrak (Month-to-month/1 Tahun/2 Tahun), Tagihan Elektronik (Paperless), Metode Pembayaran, Lama Langganan (Tenure), Biaya Bulanan (Monthly Charges), dan Total Biaya (Total Charges).
