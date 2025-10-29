# Website Soft Computing dengan Python Backend

Website interaktif untuk mempelajari tiga metode utama dalam Soft Computing:
- **Logika Fuzzy (Fuzzy Logic)** - Menangani ketidakpastian dengan derajat keanggotaan
- **Jaringan Saraf Tiruan (ANN)** - Model komputasi yang meniru neuron biologis
- **Algoritma Genetika (GA)** - Optimasi berdasarkan proses evolusi alamiah

## � Quick Start

**Ringkas:**
1. Install dependencies: `pip install -r backend/requirements.txt`
2. Jalankan backend: Double-click `backend/start_server.bat`
3. Buka website: Double-click `bungkus/index.html`
4. Coba demo di section **Materi**!

## 📁 Struktur Proyek

```
web-soft-computing/
├── backend/               # Python Flask Backend
│   ├── server.py          # API endpoints (Fuzzy, ANN, GA)
│   ├── run_server.py      # Server runner
│   ├── start_server.bat   # Quick launcher ⭐
│   └── requirements.txt   # Dependencies
├── bungkus/               # Frontend (HTML, CSS, JS)
│   ├── index.html         # Halaman utama
│   ├── style.css          # Tema abu-abu/hitam
│   └── Main.js            # Interaktivitas & API calls
├── CARA_JALANKAN.md       # 📖 Panduan lengkap
└── README.md              # Dokumentasi ini
```
```

## 📚 Penjelasan Setiap Metode

### 1. Logika Fuzzy (Fuzzy Logic)

**Konsep:** Menangani ketidakpastian dengan derajat keanggotaan (0–1) alih-alih nilai Boolean.

**Demo di Website:**
- Input: Suhu (°C) dan Kelembaban (%)
- Proses: 
  - Fuzzifikasi: Suhu → {dingin, normal, panas}, Kelembaban → {kering, sedang, lembab}
  - Inferensi: Aturan fuzzy (contoh: "Jika suhu normal DAN kelembaban sedang, maka nyaman tinggi")
  - Defuzzifikasi: Hitung skor kenyamanan 0–100
- Output: Skor kenyamanan + label (Sangat Nyaman / Cukup Nyaman / Kurang Nyaman)

**Backend (`server.py`):**
```python
@app.route('/api/fuzzy', methods=['POST'])
def fuzzy_demo():
    # Terima input suhu & kelembaban
    # Hitung membership functions
    # Jalankan fuzzy inference
    # Return skor kenyamanan
```

### 2. Jaringan Saraf Tiruan (ANN)

**Konsep:** Model komputasi yang meniru cara kerja neuron biologis untuk belajar pola dari data.

**Demo di Website:**
- Input: Dua nilai numerik (x1, x2)
- Proses:
  - Forward propagation melalui 3 layer: Input (2 neuron) → Hidden (3 neuron) → Output (1 neuron)
  - Aktivasi sigmoid di setiap neuron
  - Bobot sudah pre-trained (untuk demo)
- Output: Nilai prediksi (0–1)

**Backend (`server.py`):**
```python
class SimpleANN:
    def forward(self, inputs):
        # Hidden layer: aktivasi sigmoid
        # Output layer: aktivasi sigmoid
        # Return prediksi
```

### 3. Algoritma Genetika (Genetic Algorithm)

**Konsep:** Algoritma optimasi yang meniru proses evolusi alamiah (seleksi, crossover, mutasi).

**Demo di Website:**
- Input: String target (contoh: "HELLO") dan jumlah generasi maksimal
- Proses:
  1. Inisialisasi populasi string random
  2. Evaluasi fitness (jumlah karakter yang cocok)
  3. Seleksi individu terbaik
  4. Crossover (perkawinan dua parent)
  5. Mutasi acak
  6. Ulangi sampai target ditemukan atau mencapai generasi maksimal
- Output: String hasil evolusi + jumlah generasi yang dibutuhkan

**Backend (`server.py`):**
```python
class GeneticAlgorithm:
    def run(self, max_generations):
        # Loop setiap generasi
        # Evaluasi fitness
        # Seleksi, crossover, mutasi
        # Return hasil evolusi
```

## � Teknologi

**Frontend:**
- HTML5, CSS3 (Custom Properties, Flexbox, Grid)
- Vanilla JavaScript (ES6+, Fetch API)

**Backend:**
- Python 3.8+
- Flask (Web framework)
- Flask-CORS (Cross-Origin Resource Sharing)
- NumPy (Operasi numerik untuk ANN)

## 🎨 Fitur

- ✅ Tema abu-abu/hitam modern dengan gradien
- ✅ Responsive design (hamburger menu untuk mobile)
- ✅ Smooth scrolling antar section
- ✅ Active link highlighting saat scroll
- ✅ Interactive demos dengan real-time processing
- ✅ Error handling & loading states

## 🚀 Pengembangan Lanjutan

Ide untuk ekspansi proyek:

- [ ] Visualisasi grafik (Chart.js / D3.js)
- [ ] History hasil di localStorage
- [ ] Training ANN dengan dataset real
- [ ] Metode tambahan (PSO, ACO, Simulated Annealing)
- [ ] Deploy backend (Render, Railway, Vercel Serverless)
- [ ] Unit tests & integration tests

## 📄 Lisensi

Proyek pembelajaran untuk demonstrasi Soft Computing.

---

**Dibuat dengan ❤️ untuk pembelajaran Soft Computing**
