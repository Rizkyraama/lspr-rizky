# Setup environment

Dua environment terpisah (versi Python berbeda). Jalankan dari folder `SKRIPSI/`.

## 1. `.venv` (Python 3.14) -> `Code/Skripsi.ipynb`
Training, download Roboflow, evaluasi YOLO.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\activate
# torch CUDA 12.6 DULU (tidak ada di requirements karena versinya +cu126)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-skripsi.txt
```

## 2. `.venv312` (Python 3.12) -> `Code/Pipeline.ipynb`
PaddleOCR + inferensi (PaddleOCR belum cocok dengan Python 3.14).

```powershell
py -3.12 -m venv .venv312
.venv312\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-pipeline.txt
```

## 3. API key Roboflow
Buat file `.env` di folder `SKRIPSI/`:
```
ROBOFLOW_API_KEY=isi_key_kamu
```
`.env` sudah di `.gitignore`, jangan dibagikan.

## Cek GPU
```python
import torch; print(torch.cuda.is_available())
```
