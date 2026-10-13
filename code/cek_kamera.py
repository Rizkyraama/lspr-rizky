#!/usr/bin/env python3
"""
cek_kamera.py - pratinjau kamera dengan kotak deteksi plat, untuk mengatur posisi sebelum
menjalankan slpr.py. Skrip ini TIDAK mencatat event dan tidak menulis ke database.

Contoh (dari folder proyek, .venv312 aktif):
  python Code\\cek_kamera.py --model runs\\slpr_v8n_gabungan_run1\\weights\\best.onnx

Tombol (klik jendela pratinjau dulu supaya tombol terbaca):
  q atau Esc : keluar
  s          : simpan frame ke folder --out
  o          : baca plat pada frame ini dengan PaddleOCR (pertama kali agak lama karena memuat model)

Arti warna kotak: hijau = keyakinan deteksi >= 0.60, oranye = lebih rendah. Posisi yang baik adalah
kotak hijau yang stabil (tidak berkedip) dan plat terlihat jelas, tanpa pantulan atau blur.
"""
import argparse
import time
from pathlib import Path

import cv2

import slpr

WIN = "Pratinjau SLPR (q = keluar, s = simpan, o = baca plat)"


def best_box(boxes):
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) * b[4]) if boxes else None


def annotate(frame, boxes, extra=None):
    """Gambar kotak dan keterangan di salinan frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    for x1, y1, x2, y2, c in boxes:
        color = (0, 200, 0) if c >= 0.6 else (0, 140, 255)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{c:.2f}  lebar {int(x2 - x1)}px ({(x2 - x1) / w * 100:.0f}%)"
        cv2.putText(vis, label, (int(x1), max(int(y1) - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    status = "plat terdeteksi" if boxes else "plat TIDAK terdeteksi"
    cv2.putText(vis, f"{w}x{h}  {status}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 200, 0) if boxes else (0, 0, 255), 2)
    if extra:
        cv2.putText(vis, extra, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return vis


def run(cap, detect, read_text_factory, out_dir, show=cv2.imshow, wait=cv2.waitKey, max_frames=0):
    """Loop pratinjau. show/wait bisa diganti (untuk uji). Mengembalikan jumlah frame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader, extra, extra_until, n, n_save = None, None, 0.0, 0, 0
    gui = True
    last_snap = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("gagal membaca kamera (dipakai aplikasi lain? coba --source 1)")
            break
        n += 1
        boxes = detect(frame)
        vis = annotate(frame, boxes, extra if time.time() < extra_until else None)
        key = -1
        if gui:
            try:
                show(WIN, vis)
                key = wait(1) & 0xFF
            except cv2.error:
                gui = False
                print("jendela pratinjau tidak tersedia (OpenCV tanpa GUI). Gambar ditulis ke "
                      f"{out_dir / 'kamera_snap.jpg'} tiap detik; buka file itu. Hentikan dengan Ctrl+C.")
        if not gui and time.time() - last_snap >= 1.0:
            last_snap = time.time()
            cv2.imwrite(str(out_dir / "kamera_snap.jpg"), vis)
            b = best_box(boxes)
            print(f"{vis.shape[1]}x{vis.shape[0]}  " + (f"plat conf={b[4]:.2f} lebar={int(b[2] - b[0])}px"
                                                       if b else "plat tidak terdeteksi"))
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            n_save += 1
            p = out_dir / f"snap_{n_save:03d}.jpg"
            cv2.imwrite(str(p), frame)
            print("disimpan:", p)
        if key == ord("o"):
            b = best_box(boxes)
            if b is None:
                extra, extra_until = "tidak ada plat untuk dibaca", time.time() + 3
            else:
                if reader is None:
                    print("memuat PaddleOCR ...")
                    reader = read_text_factory()
                crop = slpr.crop_box(frame, b[:4], 0.1)
                read, frag, cands = slpr.interpret(reader(slpr.prep_crop(crop, 640)))
                txt = slpr.format_read(read) or (f"{frag['prefix']}{frag['digits']} (fragmen)" if frag else None)
                print("baris terbaca:", [(t, round(s, 2)) for t, _, s in cands], "->", txt)
                extra, extra_until = f"OCR: {txt or '(tidak terbaca)'}", time.time() + 4
        if max_frames and n >= max_frames:
            break
    return n


def main():
    ap = argparse.ArgumentParser(description="Pratinjau kamera + deteksi plat (tanpa mencatat event)")
    ap.add_argument("--model", required=True, help="path model YOLO (.onnx atau .pt)")
    ap.add_argument("--source", default="0", help="indeks webcam (0, 1, ...) atau alamat RTSP")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--yolo-threads", type=int, default=0)
    ap.add_argument("--lebar", type=int, default=0, help="minta resolusi kamera (mis. 1280); 0 = bawaan")
    ap.add_argument("--tinggi", type=int, default=0, help="minta resolusi kamera (mis. 720); 0 = bawaan")
    ap.add_argument("--out", default="hasil/kamera", help="folder untuk tombol s dan mode tanpa jendela")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if args.lebar:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.lebar)
    if args.tinggi:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.tinggi)
    if not cap.isOpened():
        print("kamera tidak bisa dibuka. Tutup aplikasi Kamera/Zoom/Teams, atau coba --source 1")
        return
    detect = slpr.make_detector(args.model, args.conf, args.imgsz, "cpu", args.yolo_threads)
    try:
        run(cap, detect, lambda: slpr.make_reader(), args.out)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
