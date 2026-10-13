#!/usr/bin/env python3
"""
Simpan crop plat dari sebuah video supaya bisa dilihat dengan mata.

Untuk tiap frame dan tiap deteksi disimpan tiga file:
  f<frame>_<k>_kotak.jpg   frame penuh dengan kotak deteksi YOLO (hijau)
  f<frame>_<k>_m00.jpg     crop persis di dalam kotak YOLO (tanpa margin)
  f<frame>_<k>_m30.jpg     crop dengan margin 30% (seperti yang dipakai OCR)

Contoh (jalankan dari folder SKRIPSI, venv .venv312 aktif):
  python Code\\lihat_crop.py --video video_uji\\parkir_1.mp4 --model runs\\slpr_v8n_gabungan_run1\\weights\\best.onnx --dari 285 --sampai 325 --out lihat_event2

Pastikan slpr.py ada di folder yang sama dengan skrip ini.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slpr  # noqa: E402


def main():
    """Baca video, jalankan detektor, simpan frame berkotak dan crop (margin 0 dan 0.3) tiap deteksi."""
    ap = argparse.ArgumentParser(description="Simpan crop plat dari video untuk dilihat")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dari", type=int, default=0, help="frame awal")
    ap.add_argument("--sampai", type=int, default=100000, help="frame akhir")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="lihat_crop")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detect = slpr.make_detector(args.model, args.conf, 640, "cpu")

    cap = cv2.VideoCapture(args.video)
    i, n_file = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or i > args.sampai:
            break
        if i >= args.dari and (i - args.dari) % args.stride == 0:
            for k, (*box, conf) in enumerate(detect(frame)):
                x1, y1, x2, y2 = [int(v) for v in box]
                vis = frame.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(str(out / f"f{i:04d}_{k}_kotak.jpg"), vis)
                for m in (0.0, 0.3):
                    crop = slpr.crop_box(frame, box, m)
                    if crop.size:
                        cv2.imwrite(str(out / f"f{i:04d}_{k}_m{int(m * 100):02d}.jpg"), crop)
                n_file += 3
                print(f"frame {i}: kotak {x1},{y1},{x2},{y2} conf={conf:.2f}")
        i += 1
    cap.release()
    print(f"selesai: {n_file} file di {out.resolve()}")


if __name__ == "__main__":
    main()
