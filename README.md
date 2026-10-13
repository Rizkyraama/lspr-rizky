# LSPR: Pengenalan Plat Nomor (YOLOv8 + PaddleOCR)

Dataset dan kode program untuk pengabdian CCTV. Deteksi plat nomor Indonesia dengan **YOLOv8n**, pembacaan teks dengan **PaddleOCR**, hasil dicatat ke SQLite (opsional integrasi Shinobi NVR).

## Isi
- `code/slpr.py` : pipeline utama (deteksi + OCR + log), bisa untuk laptop / Raspberry Pi
- `code/Skripsi.ipynb` : training dan evaluasi YOLO
- `code/Pipeline.ipynb` : percobaan pipeline YOLO + PaddleOCR
- `code/cek_kamera.py`, `code/lihat_crop.py`, `code/uji_variasi_ocr.py` : alat bantu
- `dataset/dataset_roboflow` : kamera jalan (train 1098, valid 167, test 54)
- `dataset/dataset_gabungan` : kamera jalan + Aldian terfilter (train 1744, valid 355), format YOLO, 1 kelas `plate`
- `requirements-*.txt`, `INSTALL.md` : setup environment (Python 3.14 untuk training, 3.12 untuk pipeline/PaddleOCR)

Bobot model (`*.pt`) tidak disertakan. Latih sendiri lewat `code/Skripsi.ipynb` atau minta ke pemilik repo.

## Sumber dan lisensi dataset (CC BY 4.0)
- ali-mustofa-cxrgi, *License-Plate-Indonesia*: https://universe.roboflow.com/ali-mustofa-cxrgi/license-plate-indonesia
- Aldians Workspace, *indonesian-plate-number-mpgfj-npmds* v1: https://universe.roboflow.com/aldians-workspace/indonesian-plate-number-mpgfj-npmds/dataset/1

Sebagian gambar dataset Aldian tampak berasal dari web (ada watermark), jadi lisensi CC BY pembuat dataset belum tentu mencakup hak atas foto aslinya. Sebutkan sumber jika dipakai ulang.

## Catatan
- Jalankan skrip dari folder `code/`; sesuaikan path dataset/bobot jika perlu.
- API key Roboflow disimpan di `.env` (`ROBOFLOW_API_KEY=...`), jangan di-commit.
