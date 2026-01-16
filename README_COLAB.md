# 🚀 Panduan Running di Google Colab

## Energy Sector Multi-Stock Decision Support System

**ARIMA + LSTM + Ensemble + Profit Simulation**

---

## 📋 Persiapan

### 1️⃣ Siapkan File Project

Ada 2 cara untuk upload ke Colab:

#### **CARA A: Upload ZIP (Mudah)**

1. **Zip folder project** ini menjadi `TugasAkhir.zip`
2. Pastikan struktur ZIP:
   ```
   TugasAkhir.zip
   ├── main.py
   ├── requirements.txt
   ├── src/
   │   ├── __init__.py
   │   ├── data_loader.py
   │   ├── preprocessing.py
   │   ├── decision.py
   │   ├── profit_simulation.py
   │   ├── visualization.py
   │   └── model/
   │       ├── __init__.py
   │       ├── arima_model.py
   │       ├── lstm_model.py
   │       └── ensemble.py
   └── dataset/
       └── daily/
           ├── ADRO.csv
           ├── BYAN.csv
           ├── ADMR.csv
           └── ... (semua CSV)
   ```

#### **CARA B: Google Drive (Recommended)**

1. Upload **seluruh folder** `TugasAkhir` ke Google Drive
2. Lokasi contoh: `My Drive/TugasAkhir/`

---

## 🎯 Langkah-langkah Running

### Step 1: Buka Colab

1. Buka [Google Colab](https://colab.research.google.com/)
2. Upload file `run_colab.ipynb` atau buat notebook baru

### Step 2: Install Dependencies

```python
!pip install -q pandas numpy matplotlib scikit-learn tensorflow pmdarima ta
```

### Step 3: Upload Files

**Jika pakai CARA A (ZIP):**

```python
from google.colab import files
import zipfile

uploaded = files.upload()  # Pilih TugasAkhir.zip

for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')
```

**Jika pakai CARA B (Google Drive):**

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/TugasAkhir')
```

### Step 4: Run Analysis

```python
from src.data_loader import load_stock
from src.preprocessing import add_indicators
from src.decision import make_decision
from src.profit_simulation import simulate
from src.visualization import plot_stock
from src.model.arima_model import arima_forecast
from src.model.lstm_model import lstm_forecast
from src.model.ensemble import ensemble

import pandas as pd

# Pilih saham
energy = ["ADRO","BYAN","ADMR","GEMS","ITMG","PTBA","PGAS","TCPI","DSSA","MEDC"]

results = []
for code in energy:
    df = load_stock(f"dataset/daily/{code}.csv")
    df = add_indicators(df)
    prices = df['close'].values

    a = arima_forecast(prices)
    l = lstm_forecast(prices)
    e = ensemble(a, l)

    decision, change = make_decision(prices[-1], e)

    results.append({
        "Stock": code,
        "Current": prices[-1],
        "Forecast": e,
        "Change%": change,
        "Decision": decision
    })

    plot_stock(df, e, code)

# Show results
print(pd.DataFrame(results))
profit, roi = simulate(results)
print(f"Profit: {profit}, ROI: {roi}%")
```

---

## ⚙️ Pilihan Konfigurasi

### Jumlah Saham:

**Quick Test (4 saham) - ~3 menit:**

```python
energy = ["ADRO","PTBA","ITMG","MEDC"]
```

**Recommended (10 saham) - ~7 menit:**

```python
energy = ["ADRO","BYAN","ADMR","GEMS","ITMG","PTBA","PGAS","TCPI","DSSA","MEDC"]
```

**Full Analysis (15 saham) - ~15 menit:**

```python
energy = ["ADRO","BYAN","ADMR","GEMS","ITMG","PTBA","PGAS","TCPI","DSSA","MEDC","AKRA","MCOL","BUMI","INDY","BSSR"]
```

---

## 📊 Output yang Dihasilkan

1. **Visualisasi** - Chart untuk setiap saham (inline di notebook)
2. **Tabel Hasil** - DataFrame dengan kolom:
   - Stock (kode saham)
   - Current (harga saat ini)
   - Forecast (prediksi harga)
   - Change% (persentase perubahan)
   - Decision (BUY/SELL/HOLD)
3. **Profit Simulation** - Total profit dan ROI

---

## ⚠️ Tips & Troubleshooting

### 1. **Memory Error**

Jika Colab kehabisan memory:

- Kurangi jumlah saham (gunakan Top 4)
- Restart runtime: `Runtime > Restart Runtime`
- Upgrade ke Colab Pro (lebih banyak RAM)

### 2. **File Not Found**

Pastikan:

- Struktur folder benar
- Path ke dataset/daily/\*.csv valid
- Jalankan `!ls -la` untuk cek file

### 3. **Import Error**

```python
# Cek apakah semua library terinstall
!pip list | grep -E "pandas|numpy|tensorflow|pmdarima|ta"
```

### 4. **TensorFlow Warning**

Warning TensorFlow (tentang CUDA/GPU) bisa diabaikan, tidak mempengaruhi hasil.

### 5. **Proses Lama**

- ARIMA + LSTM memang memakan waktu
- Untuk 10 saham: tunggu ~5-10 menit
- Progress akan ditampilkan untuk setiap saham

---

## 💡 Alternatif: Copy-Paste ke Colab

Jika tidak mau repot upload file, bisa copy-paste langsung:

1. Buat notebook baru di Colab
2. Buat cells untuk:
   - Cell 1: Install packages
   - Cell 2-10: Copy isi setiap file .py ke cells terpisah
   - Cell 11: Upload CSV manual
   - Cell 12: Run main code

---

## 📞 Support

Jika ada kendala:

1. Cek error message di output cell
2. Pastikan semua file ter-upload lengkap
3. Verify dependencies terinstall
4. Restart runtime dan coba lagi

---

## ✅ Checklist Sebelum Running

- [ ] File `run_colab.ipynb` sudah di-upload
- [ ] Dependencies ter-install
- [ ] Folder struktur benar (src/, dataset/)
- [ ] File CSV saham tersedia
- [ ] Pilih jumlah saham yang akan dianalisa
- [ ] Colab runtime active

**Selamat Menganalisa! 🚀📈**
