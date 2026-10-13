#!/usr/bin/env python3
"""
Bandingkan beberapa variasi OCR pada crop plat yang sudah disimpan (hasil lihat_crop.py).

Variasi:
  asli          pipeline sekarang (lebar 640, PaddleOCR penuh)
  besar         lebar 1280
  kontras       CLAHE (menaikkan kontras) lalu PaddleOCR penuh
  longgar       ambang deteksi teks diturunkan (nama argumen belum diuji di versi Anda)
  baris_atas    hanya bagian atas crop (baris utama plat) langsung ke pembaca teks,
                tanpa tahap deteksi teks. Mengandalkan tata letak plat: baris utama di atas,
                masa berlaku di bawah. Cocok untuk crop persis di kotak YOLO (_m00).
  baris_atas_k  seperti baris_atas, dengan CLAHE

Contoh (dari folder SKRIPSI, venv .venv312 aktif):
  python Code\\uji_variasi_ocr.py --folder lihat_event2 --pola "*_m00.jpg" --benar N3869L
  python Code\\uji_variasi_ocr.py --selftest
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slpr  # noqa: E402


def bersih(t):
    """Huruf besar, hanya A-Z dan 0-9."""
    return re.sub(r"[^A-Z0-9]", "", str(t).upper())


def kontras(img):
    """Naikkan kontras dengan CLAHE pada kanal L (ruang warna LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def jalankan(img, mesin, rasio_atas=0.65):
    """mesin: dict dengan fungsi 'ocr' (img -> [(teks, skor, poligon)]), 'longgar' (boleh None)
    dan 'rec' (img -> [(teks, skor)]). Mengembalikan {nama_variasi: (ringkasan, [teks_bersih])}."""
    hasil = {}

    def penuh(nama, fn, gambar):
        read, cands = slpr.read_plate(slpr.to_items(fn(gambar)))
        baris = [bersih(t) for t, _, _ in cands]
        hasil[nama] = (f"{slpr.format_read(read) or '-'}   baris terbaca: {baris}", baris)

    def rec_saja(nama, gambar):
        teks = [(bersih(t), s) for t, s in mesin["rec"](gambar)]
        hasil[nama] = (" | ".join(f"{t}({s:.2f})" for t, s in teks), [t for t, _ in teks])

    p640 = slpr.prep_crop(img, 640)
    penuh("asli", mesin["ocr"], p640)
    penuh("besar", mesin["ocr"], slpr.prep_crop(img, 1280))
    penuh("kontras", mesin["ocr"], kontras(p640))
    if mesin.get("longgar"):
        penuh("longgar", mesin["longgar"], p640)
    atas = img[: max(int(img.shape[0] * rasio_atas), 1)]
    rec_saja("baris_atas", slpr.prep_crop(atas, 640))
    rec_saja("baris_atas_k", kontras(slpr.prep_crop(atas, 640)))
    return hasil


def buat_ocr(**kw):
    """PaddleOCR penuh (deteksi + baca teks); kw = argumen tambahan PaddleOCR."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False, **kw)

    def baca(img):
        raw = []
        for r in ocr.predict(img):
            raw += list(zip(r.get("rec_texts", []), r.get("rec_scores", []), r.get("rec_polys", [])))
        return raw
    return baca


def buat_rec():
    """Pembaca teks saja (tanpa deteksi teks); None bila tidak ada model yang bisa dimuat."""
    from paddleocr import TextRecognition
    rec = None
    for nama in ("PP-OCRv6_medium_rec", "en_PP-OCRv5_mobile_rec"):
        try:
            rec = TextRecognition(model_name=nama)
            print("pembaca teks:", nama)
            break
        except Exception as e:
            print(f"model {nama} gagal dimuat: {str(e)[:100]}")
    if rec is None:
        return None

    def baca(img):
        return [(r.get("rec_text", ""), float(r.get("rec_score", 0))) for r in rec.predict(input=img)]
    return baca


def selftest():
    """Uji tanpa model: mesin palsu harus menghasilkan semua variasi tanpa longgar."""
    img = np.full((40, 120, 3), 90, np.uint8)
    poly = [[0, 0], [100, 0], [100, 30], [0, 30]]
    mesin = {
        "ocr": lambda im: [("N 3869 L", 0.9, poly)],
        "longgar": None,
        "rec": lambda im: [("N3869L", 0.95)],
    }
    h = jalankan(img, mesin)
    assert set(h) == {"asli", "besar", "kontras", "baris_atas", "baris_atas_k"}, set(h)
    assert "N3869L" in h["baris_atas"][1] and h["asli"][1] == ["N3869L"], h
    print("SELFTEST OK")


def main():
    """Jalankan semua variasi OCR pada tiap crop di --folder dan tampilkan ringkasan kecocokan."""
    ap = argparse.ArgumentParser(description="Bandingkan variasi OCR pada crop plat")
    ap.add_argument("--folder")
    ap.add_argument("--pola", default="*_m00.jpg")
    ap.add_argument("--benar", default="", help="plat yang benar tanpa spasi, mis. N3869L")
    ap.add_argument("--rasio-atas", type=float, default=0.65,
                    help="porsi tinggi crop yang dianggap baris utama plat")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.folder:
        ap.error("--folder wajib diisi")

    files = sorted(Path(args.folder).glob(args.pola))
    if not files:
        sys.exit(f"tidak ada file {args.pola} di {args.folder}")
    ocr = buat_ocr()
    longgar = None
    try:
        longgar = buat_ocr(text_det_thresh=0.2, text_det_box_thresh=0.4)
    except Exception as e:
        print("variasi 'longgar' dilewati (argumen tidak dikenal?):", str(e)[:120])
    rec = buat_rec()
    if rec is None:
        sys.exit("pembaca teks tidak bisa dimuat")
    mesin = {"ocr": ocr, "longgar": longgar, "rec": rec}

    benar = bersih(args.benar)
    hit, total = {}, 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        total += 1
        hasil = jalankan(img, mesin, args.rasio_atas)
        print(f"\n{f.name}  ({img.shape[1]}x{img.shape[0]} px)")
        for nama, (ringkas, teks) in hasil.items():
            ok = bool(benar) and any(benar in t for t in teks)
            hit[nama] = hit.get(nama, 0) + int(ok)
            print(f"  {nama:13s}: {ringkas}   {'<-- COCOK' if ok else ''}")
    if benar:
        print(f"\nRingkasan: berapa gambar yang mengandung '{benar}' (dari {total})")
        for nama, n in hit.items():
            print(f"  {nama:13s}: {n}/{total}")


if __name__ == "__main__":
    main()
