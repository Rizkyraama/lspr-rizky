#!/usr/bin/env python3
"""
SLPR - Smart License Plate Recognition
YOLOv8 (deteksi plat) + PaddleOCR (baca teks) + SQLite (catatan) + log kinerja.

Satu skrip yang sama dipakai di laptop dan di Raspberry Pi.
Sumber (--source) bisa: folder foto, satu foto, file video, indeks webcam (0),
atau alamat RTSP.

Contoh:
  python slpr.py --selftest
  python slpr.py --source ../foto_uji --model ../runs/slpr_v8n_gabungan_run1/weights/best.onnx
  python slpr.py --source video_uji.mp4 --model best.onnx --stride 5
  python slpr.py --source rtsp://127.0.0.1:8554/cam --model best.onnx --stride 5

Cara membaca plat (versi ini):
  - Tiap pembacaan dipecah jadi INTI (angka + huruf seri, mis. 5285KF) dan AWALAN
    (huruf wilayah, mis. N). Awalan sering tidak terbaca di satu frame, sedangkan
    inti biasanya konsisten di banyak frame.
  - Satu sesi kendaraan memilih INTI dengan suara terbanyak, lalu AWALAN dari
    pembacaan lengkap yang inti-nya sama. Jika tidak pernah ada awalan, plat
    dicatat kosong dan intinya disimpan di kolom "parsial" (tidak ditebak).
  - FRAGMEN: bacaan yang hanya berisi awalan + angka (mis. "N5285", tanpa huruf seri)
    tidak dibuang. Awalannya ikut dihitung HANYA bila angkanya sama dengan angka inti
    pemenang sesi (mis. inti 5285KF + fragmen N5285 -> "N 5285 KF"). Fragmen dengan
    angka lain (mis. baris masa berlaku "C930") diabaikan. Jumlah suara awalan yang
    berasal dari fragmen dicatat di kolom suara_fragmen.

Sumber langsung (RTSP/webcam):
  - Frame dibaca di thread terpisah dan hanya frame terbaru yang diproses.
  - OCR (lambat) dijalankan di thread pekerja terpisah (--ocr-mode thread, otomatis
    untuk sumber langsung), sehingga deteksi tetap jalan saat OCR berlangsung. Hanya
    SATU crop menunggu giliran OCR (yang terbaru); crop yang tertimpa dihitung di
    kolom ocr_dilewati. Jumlah bacaan per kendaraan tetap dibatasi kecepatan OCR.

Batasan versi ini:
  - Plat tanpa huruf seri atau dengan 1 angka (jarang) tidak akan terbaca.
  - Satu "sesi kendaraan" diproses sekali waktu (kendaraan bersamaan dalam satu
    frame akan digabung ke satu sesi).
  - Mode thread OCR dan pembaca frame baru diuji dengan stream sintetis dan
    sumber palsu; tes di laptop dan Raspberry Pi masih berjalan.
  - Pengiriman event ke Shinobi (--shinobi) belum diuji pada Shinobi sungguhan.
"""
import argparse
import csv
import json
import os
import platform
import re
import sqlite3
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil
except ImportError:  # opsional
    psutil = None

# awalan (0-2 huruf wilayah), angka (2-4), huruf seri (1-3, wajib), sisa angka (masa berlaku) dibuang
CORE = re.compile(r"^([A-Z]{0,2})(\d{2,4})([A-Z]{1,3})(\d*)$")
# fragmen: awalan (1-2 huruf) + angka (2-4), tanpa huruf seri; sisa angka dibuang
FRAG = re.compile(r"^([A-Z]{1,2})(\d{2,4})(\d*)$")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------- pengukuran
def cpu_temp():
    """Suhu CPU (derajat C) di Linux/Raspberry Pi, None di sistem lain."""
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except Exception:
        return None


# ------------------------------------------------------------ pembacaan plat
def to_items(raw):
    """raw: daftar (teks, skor, poligon) dari mesin OCR -> daftar dict bersih."""
    items = []
    for t, s, poly in raw:
        t = re.sub(r"[^A-Z0-9]", "", str(t).upper())
        if not t:
            continue
        p = np.asarray(poly, dtype=float)
        y1, y2 = p[:, 1].min(), p[:, 1].max()
        items.append({"t": t, "s": float(s), "x": float(p[:, 0].min()),
                      "yc": (y1 + y2) / 2, "h": y2 - y1})
    return items


def read_plate(items, ytol=0.6):
    """Kelompokkan potongan teks per baris (kiri ke kanan). Dari baris yang berbentuk
    inti plat (angka + huruf seri) ambil yang tertinggi (baris masa berlaku lebih kecil).
    Mengembalikan (read, baris): read = {"prefix","digits","suffix","core"} atau None;
    baris = [(teks, tinggi, skor_ocr), ...]."""
    if not items:
        return None, []
    items = sorted(items, key=lambda d: d["yc"])
    lines, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it["yc"] - cur[-1]["yc"]) < ytol * max(it["h"], cur[-1]["h"]):
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)
    cands = []
    for ln in lines:
        ln.sort(key=lambda d: d["x"])
        cands.append(("".join(d["t"] for d in ln), sum(d["h"] for d in ln) / len(ln),
                      sum(d["s"] for d in ln) / len(ln)))
    cores = [(h, CORE.match(txt)) for txt, h, _ in cands if CORE.match(txt)]
    if not cores:
        return None, cands
    _, m = max(cores, key=lambda x: x[0])
    prefix, digits, suffix, _ = m.groups()
    return {"prefix": prefix, "digits": digits, "suffix": suffix, "core": digits + suffix}, cands


def read_fragment(cands):
    """Dipakai bila read_plate() tidak menemukan inti. Dari baris berbentuk awalan + angka
    (mis. "N5285") ambil yang tertinggi. Mengembalikan {"prefix","digits"} atau None."""
    best = None
    for txt, h, _ in cands:
        m = FRAG.match(txt)
        if m and (best is None or h > best[0]):
            best = (h, {"prefix": m.group(1), "digits": m.group(2)})
    return best[1] if best else None


def interpret(raw):
    """Keluaran mentah OCR -> (read, frag, baris). frag hanya diisi bila read kosong."""
    read, cands = read_plate(to_items(raw))
    return read, (None if read else read_fragment(cands)), cands


def format_read(read):
    if not read:
        return None
    if read["prefix"]:
        return f"{read['prefix']} {read['digits']} {read['suffix']}"
    return read["core"]


def choose_plate(items, ytol=0.6):
    """Pembungkus lama: (plat, lengkap, baris). lengkap = ada awalan wilayah."""
    read, cands = read_plate(items, ytol)
    if read:
        return format_read(read), bool(read["prefix"]), cands
    if not cands:
        return None, False, []
    return max(cands, key=lambda c: c[1])[0], False, cands


def crop_box(frame, box, margin=0.3):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    mx, my = int(margin * (x2 - x1)), int(margin * (y2 - y1))
    return frame[max(y1 - my, 0):min(y2 + my, H), max(x1 - mx, 0):min(x2 + mx, W)]


def prep_crop(crop, width=640):
    s = width / crop.shape[1]
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=interp)
    return cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_REPLICATE)


# ------------------------------------------------------------- sesi kendaraan
class Session:
    """Mengumpulkan pembacaan dari beberapa frame lalu memilih inti plat dengan suara
    terbanyak dan awalan dari pembacaan lengkap. Sesi berakhir jika tidak ada
    deteksi selama gap_s detik."""

    def __init__(self, gap_s=1.5):
        self.gap_s = gap_s
        self.reset()

    def reset(self):
        self.reads, self.n_det = [], 0          # reads: [(read, conf_deteksi)]
        self.frags = []                         # frags: [(fragmen, conf_deteksi)]
        self.best_conf, self.best_crop, self.best_box, self.last_t = 0.0, None, None, None
        self.any_conf, self.any_crop, self.any_box = 0.0, None, None   # terbaik walau tidak terbaca

    def active(self):
        return self.last_t is not None

    def expired(self, t):
        return self.active() and (t - self.last_t) > self.gap_s

    def add_det(self, t, det_conf, crop, box):
        """Catat satu deteksi (OCR-nya boleh menyusul lewat add_read)."""
        self.last_t = t
        self.n_det += 1
        if crop is not None and det_conf >= self.any_conf:
            self.any_conf, self.any_crop, self.any_box = det_conf, crop, box

    def add_read(self, read, det_conf, crop, box, frag=None):
        """Catat hasil OCR: bacaan berinti (read) atau fragmen awalan + angka (frag)."""
        if read:
            self.reads.append((read, det_conf))
            if det_conf >= self.best_conf:
                self.best_conf, self.best_crop, self.best_box = det_conf, crop, box
        elif frag:
            self.frags.append((frag, det_conf))

    def add(self, t, read, det_conf, crop, box, frag=None):
        self.add_det(t, det_conf, crop, box)
        self.add_read(read, det_conf, crop, box, frag)

    def _summary(self):
        cores = Counter(r["core"] for r, _ in self.reads)
        plate = parsial = None
        votes = pv = fv = 0
        if cores:
            core = max(cores, key=lambda c: (cores[c], sum(cf for r, cf in self.reads if r["core"] == c)))
            votes = cores[core]
            same = [r for r, _ in self.reads if r["core"] == core]
            digits = same[0]["digits"]
            pref = Counter(r["prefix"] for r in same if r["prefix"])
            # fragmen ikut memilih awalan hanya bila angkanya sama dengan inti pemenang
            pref_frag = Counter(f["prefix"] for f, _ in self.frags if f["digits"] == digits)
            pref.update(pref_frag)
            if pref:
                prefix = max(pref, key=pref.get)
                pv, fv = pref[prefix], pref_frag.get(prefix, 0)
                plate = f"{prefix} {digits} {same[0]['suffix']}"
            else:
                parsial = core
        return {"plate": plate, "parsial": parsial, "votes": votes, "prefix_votes": pv,
                "frag_votes": fv, "n_reads": len(self.reads), "n_det": self.n_det,
                "conf_det": round(self.best_conf if self.best_crop is not None else self.any_conf, 3),
                "crop": self.best_crop if self.best_crop is not None else self.any_crop,
                "box": self.best_box if self.best_crop is not None else self.any_box}

    def ready(self, n):
        """True bila plat lengkap punya >= n suara inti DAN awalan pemenangnya juga >= n suara
        (dipakai --stop-votes). Syarat awalan mencegah berhenti terlalu dini pada awalan yang
        kebetulan salah baca satu kali (mis. N terbaca H)."""
        s = self._summary()
        return bool(s["plate"]) and s["votes"] >= n and s["prefix_votes"] >= n

    def finalize(self):
        out = self._summary()
        self.reset()
        return out


# ------------------------------------------------------------------ database
class EventDB:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, perangkat TEXT, sumber TEXT,
                plat TEXT, suara INTEGER, n_baca INTEGER, n_deteksi INTEGER,
                conf_deteksi REAL, crop TEXT, parsial TEXT, suara_awalan INTEGER,
                suara_fragmen INTEGER)""")
        have = {r[1] for r in self.con.execute("PRAGMA table_info(events)")}
        for col, typ in (("parsial", "TEXT"), ("suara_awalan", "INTEGER"), ("suara_fragmen", "INTEGER")):
            if col not in have:  # database dari versi lama
                self.con.execute(f"ALTER TABLE events ADD COLUMN {col} {typ}")
        self.con.commit()

    def count(self):
        return self.con.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def add(self, row):
        self.con.execute(
            "INSERT INTO events(ts,perangkat,sumber,plat,suara,n_baca,n_deteksi,conf_deteksi,"
            "crop,parsial,suara_awalan,suara_fragmen) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
        self.con.commit()

    def close(self):
        self.con.close()


# ------------------------------------------------------------------- Shinobi
def build_shinobi_url(base, api_key, group, monitor, plate, conf, box):
    """URL pemicu Detector Engine Shinobi (sesuai dokumentasi API Monitor Triggers)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    data = {"plug": "slpr", "name": "plat", "reason": "object", "confidence": float(conf),
            "matrices": [{"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                          "tag": plate, "confidence": float(conf)}]}
    return (f"{base.rstrip('/')}/{api_key}/motion/{group}/{monitor}?data="
            + urllib.parse.quote(json.dumps(data)))


def send_shinobi(url):
    try:
        urllib.request.urlopen(url, timeout=3).read()
        return True
    except Exception as e:  # jangan sampai kegagalan Shinobi menghentikan pipeline
        print("  [shinobi] gagal kirim:", e)
        return False


# --------------------------------------------------------------------- sumber
def is_live(source):
    s = str(source)
    return s.isdigit() or s.startswith(("rtsp://", "rtmp://", "http://"))


class LatestFrameReader:
    """Membaca stream di thread terpisah dan hanya menyimpan frame TERBARU.
    Bila pemrosesan lebih lambat dari kedatangan frame, frame yang tidak sempat
    diproses dilewati (dan dihitung), sehingga latensi tidak menumpuk."""

    def __init__(self, src, cap_factory=None):
        self.cap = (cap_factory or cv2.VideoCapture)(src)
        self._lock = threading.Lock()
        self.frame, self.seq, self.t_cap = None, 0, 0.0
        self.ended = False
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop:
            ok, fr = self.cap.read()
            if not ok:
                break
            with self._lock:
                self.frame, self.seq, self.t_cap = fr, self.seq + 1, time.time()
        self.ended = True

    def next_frame(self, last_seq):
        """Tunggu frame yang lebih baru dari last_seq. Mengembalikan (seq, frame, waktu_tangkap)
        atau None bila stream berakhir."""
        while True:
            with self._lock:
                if self.seq > last_seq:
                    return self.seq, self.frame, self.t_cap
                done = self.ended
            if done:
                return None
            time.sleep(0.002)

    def close(self):
        self._stop = True
        self._th.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


def iter_frames(source, threaded=True):
    """Menghasilkan (indeks, frame, waktu_detik, nama, per_item[, meta]).
    per_item=True untuk foto (tiap foto dianggap satu kendaraan).
    Untuk sumber langsung (RTSP/webcam) dengan threaded=True, frame dibaca di thread terpisah dan
    meta = {"live": True, "age_ms": umur frame saat diambil, "dropped": frame yang dilewati}."""
    p = Path(str(source))
    if p.is_dir() or (p.is_file() and p.suffix.lower() in IMG_EXT):
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
        for i, f in enumerate(files):
            fr = cv2.imread(str(f))
            if fr is None:
                print("gagal baca:", f.name)
                continue
            yield i, fr, float(i), f.name, True
        return
    src = int(source) if str(source).isdigit() else str(source)
    live = is_live(source)
    if live and threaded:
        rd = LatestFrameReader(src)
        last = 0
        try:
            while True:
                nxt = rd.next_frame(last)
                if nxt is None:
                    break
                seq, fr, t_cap = nxt
                meta = {"live": True, "age_ms": (time.time() - t_cap) * 1000,
                        "dropped": (seq - last - 1) if last else 0}
                last = seq
                yield seq, fr, t_cap, f"frame{seq}", False, meta
        finally:
            rd.close()
        return
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = time.time() if (live or fps <= 0) else i / fps
        yield i, fr, t, f"frame{i}", False
        i += 1
    cap.release()


# ------------------------------------------------------------------- pipeline
def flush(sess, db, cfg, source, shinobi):
    r = sess.finalize()
    k = db.count() + 1
    crop_path = ""
    if r["crop"] is not None:
        crop_path = str(Path(cfg.out) / "crops" / f"event_{k:04d}.jpg")
        cv2.imwrite(crop_path, r["crop"])
    db.add((datetime.now().isoformat(timespec="seconds"), cfg.perangkat, str(source),
            r["plate"], r["votes"], r["n_reads"], r["n_det"], r["conf_det"], crop_path,
            r["parsial"], r["prefix_votes"], r["frag_votes"]))
    print(f"[event {k}] plat={r['plate']} parsial={r['parsial']} "
          f"suara={r['votes']}/{r['n_reads']} awalan={r['prefix_votes']} "
          f"fragmen={r['frag_votes']} deteksi={r['n_det']}")
    if shinobi and r["plate"]:
        send_shinobi(build_shinobi_url(shinobi["base"], shinobi["api_key"], shinobi["group"],
                                       shinobi["monitor"], r["plate"], r["conf_det"], r["box"]))
    return r


class OcrWorker:
    """Menjalankan OCR di thread terpisah. Hanya SATU pekerjaan yang boleh menunggu
    (yang terbaru); pekerjaan lama yang tertimpa dihitung di self.skipped."""

    def __init__(self, read_text):
        self.read_text = read_text
        self._cv = threading.Condition()
        self._pending, self._busy, self._stop = None, False, False
        self._results = []
        self.skipped = 0
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def submit(self, job):
        with self._cv:
            if self._pending is not None:
                self.skipped += 1
            self._pending = job
            self._cv.notify_all()

    def _run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._pending is None:
                    return
                job, self._pending = self._pending, None
                self._busy = True
            t0 = time.time()
            try:
                raw = self.read_text(job["img"])
            except Exception as e:  # satu OCR gagal tidak boleh mematikan pipeline
                raw = []
                print("  [ocr] gagal:", e)
            res = {k: v for k, v in job.items() if k != "img"}
            res.update(raw=raw, ms=(time.time() - t0) * 1000)
            with self._cv:
                self._results.append(res)
                self._busy = False
                self._cv.notify_all()

    def wait_idle(self, timeout=60.0):
        end = time.time() + timeout
        with self._cv:
            while (self._pending is not None or self._busy) and time.time() < end:
                self._cv.wait(0.05)

    def collect(self):
        with self._cv:
            out, self._results = self._results, []
        return out

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._th.join(timeout=2.0)


def process(source, detect, read_text, cfg, db, log, shinobi=None, frames=None, reads_log=None):
    """detect(frame) -> [(x1,y1,x2,y2,conf)]; read_text(img) -> [(teks, skor, poligon)].
    cfg.ocr_mode: "sync" (OCR di loop utama), "thread" (OCR di thread pekerja), atau "auto"
    (thread hanya untuk sumber langsung RTSP/webcam)."""
    sess = Session(cfg.gap)
    mode = getattr(cfg, "ocr_mode", "auto")
    use_thread = mode == "thread" or (mode == "auto" and frames is None and is_live(source))
    worker = OcrWorker(read_text) if use_thread else None
    print("OCR:", "thread terpisah (hanya crop terbaru yang dibaca)" if use_thread else "di loop utama")
    acc = {"ocr_ms": 0.0}
    n_proc, t_start = 0, time.time()

    def apply(res):
        """Hasil OCR (sinkron maupun dari thread) -> catatan bacaan + sesi."""
        read, frag, cands = interpret(res["raw"])
        acc["ocr_ms"] += res["ms"]
        shown = format_read(read) if read else (f"frag:{frag['prefix']}{frag['digits']}" if frag else None)
        if reads_log is not None:
            reads_log.writerow([datetime.now().isoformat(timespec="seconds"), res["name"], shown,
                                bool(read and read["prefix"]),
                                json.dumps([[tx, round(h, 1), round(sc, 2)] for tx, h, sc in cands])])
        sess.add_read(read, res["conf"], res["crop"], res["box"], frag)
        if getattr(cfg, "cetak_bacaan", False):   # tampilkan tiap bacaan + perkiraan plat sesi saat ini
            sm = sess._summary()
            now = sm["plate"] or (f"{sm['parsial']} (awalan belum terbaca)" if sm["parsial"] else "-")
            print(f"  [baca {res['name']}] {shown or '(tidak terbaca)'} | sementara: {now} "
                  f"(suara {sm['votes']}, awalan {sm['prefix_votes']})")

    def settle():
        """Tunggu OCR yang masih berjalan dan masukkan hasilnya (wajib sebelum menutup sesi)."""
        if worker:
            worker.wait_idle()
            for res in worker.collect():
                apply(res)

    show = {"on": bool(getattr(cfg, "tampil", False)), "last": "", "until": 0.0}

    def do_flush():
        r = flush(sess, db, cfg, source, shinobi)
        show["last"] = f"terakhir: {r['plate'] or r['parsial'] or 'tidak terbaca'}"
        show["until"] = time.time() + 4

    def render(frame, boxes):
        """Gambar pratinjau (kotak plat + plat sementara). True bila pengguna menekan q atau Esc."""
        scale = min(1.0, 960 / frame.shape[1])
        vis = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for x1, y1, x2, y2, c in boxes:
            cv2.rectangle(vis, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), (0, 200, 0), 2)
            cv2.putText(vis, f"{c:.2f}", (int(x1 * scale), max(int(y1 * scale) - 6, 14)), font, 0.5, (0, 200, 0), 2)
        if sess.active():
            sm = sess._summary()
            line = "sesi: " + (sm["plate"] or (f"{sm['parsial']} (awalan ?)" if sm["parsial"] else "membaca ..."))
        elif time.time() < show["until"]:
            line = show["last"]
        else:
            line = "menunggu plat ..."
        cv2.putText(vis, line, (8, 24), font, 0.7, (0, 255, 255), 2)
        try:
            cv2.imshow("SLPR (q = berhenti)", vis)
            return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)
        except cv2.error:
            show["on"] = False
            print("jendela tampilan tidak tersedia (OpenCV tanpa GUI), --tampil dimatikan")
            return False

    src_iter = frames if frames is not None else iter_frames(source, not getattr(cfg, "tanpa_thread", False))
    try:
        for item in src_iter:
            i, frame, t, name, per_item = item[:5]
            meta = item[5] if len(item) > 5 else {}
            if cfg.max_frames and i >= cfg.max_frames:
                break
            t_iter = time.time()
            acc["ocr_ms"] = 0.0
            skipped0 = worker.skipped if worker else 0
            if worker:
                for res in worker.collect():
                    apply(res)
            if sess.expired(t):
                settle()
                do_flush()
            if not per_item and not meta.get("live") and i % cfg.stride:
                continue
            t0 = time.time()
            boxes = detect(frame)
            det_ms = (time.time() - t0) * 1000
            best = None
            for *box, conf in boxes:
                crop = crop_box(frame, box, cfg.margin)
                if crop.size == 0:
                    continue
                if cfg.stop_votes and sess.ready(cfg.stop_votes):
                    sess.add(t, None, conf, None, box)  # deteksi dihitung, OCR dilewati
                    continue
                if worker:
                    sess.add_det(t, conf, crop, box)
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if best is None or conf * area > best[0]:
                        best = (conf * area, crop, box, conf)
                else:
                    sess.add_det(t, conf, crop, box)
                    t1 = time.time()
                    res = {"raw": read_text(prep_crop(crop, cfg.ocr_width)), "name": name, "conf": conf,
                           "crop": crop, "box": box, "ms": (time.time() - t1) * 1000}
                    apply(res)
            if best is not None:   # thread: satu crop terbaik per frame diserahkan ke pekerja
                _, crop, box, conf = best
                worker.submit({"img": prep_crop(crop, cfg.ocr_width), "name": name, "conf": conf,
                               "crop": crop, "box": box})
            n_proc += 1
            # total_ms = waktu nyata satu putaran (deteksi + crop + OCR bila sinkron);
            # ocr_ms di mode thread = lama OCR yang selesai pada putaran ini (jalan di latar belakang)
            log.writerow([datetime.now().isoformat(timespec="seconds"), cfg.perangkat, name,
                          len(boxes), round(det_ms, 1), round(acc["ocr_ms"], 1),
                          round((time.time() - t_iter) * 1000, 1),
                          psutil.cpu_percent() if psutil else "",
                          psutil.virtual_memory().percent if psutil else "",
                          round(psutil.Process().memory_info().rss / 1e6) if psutil else "",
                          cpu_temp() if cpu_temp() is not None else "",
                          round(meta["age_ms"], 1) if "age_ms" in meta else "",
                          meta.get("dropped", ""),
                          (worker.skipped - skipped0) if worker else 0])
            if show["on"] and render(frame, boxes):
                print("\ndihentikan (tombol q), menyelesaikan sesi terakhir ...")
                break
            if per_item and sess.active():
                settle()
                do_flush()
        settle()
        if sess.active():
            do_flush()
    except KeyboardInterrupt:   # Ctrl+C: selesaikan sesi terakhir lalu berhenti rapi
        print("\ndihentikan (Ctrl+C), menyelesaikan sesi terakhir ...")
        settle()
        if sess.active():
            do_flush()
    finally:
        if worker:
            worker.close()
        if getattr(cfg, "tampil", False):
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    dur = time.time() - t_start
    print(f"selesai: {n_proc} frame diproses dalam {dur:.1f} s")
    return n_proc


# ------------------------------------------------------------ mesin sungguhan
def limit_threads(n):
    """Batasi thread CPU untuk YOLO (PyTorch dan ONNX Runtime). Tanpa ini, thread pool dibuat
    sebanyak core mesin, jadi simulasi 'Pi 4 core' di laptop berebut core dan hasilnya pesimis."""
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        import onnxruntime as ort
        orig = ort.InferenceSession.__init__

        def init(self, path_or_bytes, sess_options=None, *a, **k):
            if sess_options is None:
                sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = n
            orig(self, path_or_bytes, sess_options, *a, **k)
        ort.InferenceSession.__init__ = init
    except Exception as e:
        print("peringatan: batas thread ONNX Runtime tidak terpasang:", e)


def make_detector(model_path, conf, imgsz, device, threads=0):
    if threads:
        limit_threads(threads)
    from ultralytics import YOLO
    model = YOLO(model_path, task="detect")
    dev = int(device) if str(device).isdigit() else device

    def detect(frame):
        r = model.predict(frame, conf=conf, imgsz=imgsz, device=dev, verbose=False)[0]
        return [(*b, float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy().tolist(),
                                               r.boxes.conf.cpu().numpy().tolist())]
    detect(np.zeros((imgsz, imgsz, 3), np.uint8))  # pemanasan, agar frame pertama tidak lambat
    return detect


def parse_kw(items):
    """['k=v', ...] -> dict; nilai dibaca sebagai JSON bila bisa (0.2 -> float, true -> True)."""
    kw = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            kw[k.strip()] = json.loads(v)
        except Exception:
            kw[k.strip()] = v
    return kw


def make_reader(det_model=None, rec_model=None, extra=None, threads=0):
    from paddleocr import PaddleOCR
    kw = dict(extra or {})
    if threads:
        kw["cpu_threads"] = threads   # belum diverifikasi pada PaddleOCR 3.7; bila ditolak, hapus opsi ini
    if det_model:
        kw["text_detection_model_name"] = det_model   # belum diuji
    if rec_model:
        kw["text_recognition_model_name"] = rec_model  # belum diuji
    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False, **kw)

    def read_text(img):
        raw = []
        for r in ocr.predict(img):
            raw += list(zip(r.get("rec_texts", []), r.get("rec_scores", []), r.get("rec_polys", [])))
        return raw
    warm = np.full((120, 400, 3), 255, np.uint8)
    cv2.putText(warm, "B 1234 XY", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    read_text(warm)  # pemanasan
    return read_text


# ------------------------------------------------------------------- selftest
def selftest():
    def poly(x, y, w, h):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    def rd(prefix, digits, suffix):
        return {"prefix": prefix, "digits": digits, "suffix": suffix, "core": digits + suffix}

    # 1) masa berlaku menempel -> tetap terbaca
    items = to_items([("B6229VXD05270", 0.95, poly(0, 0, 100, 30)), ("0527", 0.9, poly(0, 40, 50, 15))])
    assert choose_plate(items)[:2] == ("B 6229 VXD", True), choose_plate(items)
    # 2) teks sampah kecil vs plat besar -> pilih plat
    items = to_items([("IN5025", 0.9, poly(0, 0, 60, 10)), ("AD2832NW", 0.9, poly(0, 30, 100, 30))])
    assert choose_plate(items)[:2] == ("AD 2832 NW", True), choose_plate(items)
    # 3) potongan terpisah dibaca urut kiri ke kanan
    items = to_items([("4127YB", 0.9, poly(40, 0, 80, 30)), ("H", 0.9, poly(0, 0, 20, 30))])
    assert choose_plate(items)[:2] == ("H 4127 YB", True), choose_plate(items)
    # 3b) tanpa huruf seri ditolak; baris masa berlaku "C9-30" ditolak
    assert choose_plate(to_items([("4127", 0.9, poly(40, 0, 60, 30)), ("H", 0.9, poly(0, 0, 20, 30))]))[:2] == ("H4127", False)
    assert choose_plate(to_items([("C9-30", 0.95, poly(0, 0, 60, 18))]))[:2] == ("C930", False)
    # 3c) tanpa awalan: inti terbaca, tidak lengkap; baris masa berlaku yang mirip plat ("C93G") kalah
    rp, _ = read_plate(to_items([("5573KC", 1.0, poly(20, 0, 100, 34)), ("C93G", 0.89, poly(20, 40, 60, 26))]))
    assert rp == rd("", "5573", "KC"), rp
    # 4) bukan plat
    assert choose_plate(to_items([("HONDA", 0.9, poly(0, 0, 80, 20))]))[:2] == ("HONDA", False)
    assert choose_plate([]) == (None, False, [])
    # 5) voting inti + awalan
    s = Session(1.0)
    for r in [rd("", "6229", "VXD"), rd("B", "6229", "VXD"), rd("", "6229", "VXD"), rd("", "6229", "VXO")]:
        s.add(0.0, r, 0.9, None, (0, 0, 1, 1))
    assert s.expired(2.0)
    f = s.finalize()
    assert (f["plate"], f["votes"], f["prefix_votes"]) == ("B 6229 VXD", 3, 1), f
    s.add(0.0, rd("", "3869", "L"), 0.9, None, (0, 0, 1, 1))  # tanpa awalan -> parsial
    f = s.finalize()
    assert (f["plate"], f["parsial"]) == (None, "3869L"), f
    # 5b) argumen tambahan OCR
    assert parse_kw(["text_det_thresh=0.2", "x=abc", "flag=true"]) == {"text_det_thresh": 0.2, "x": "abc", "flag": True}
    # 6) URL Shinobi
    u = build_shinobi_url("http://h:8080/", "KEY", "GRP", "MON", "B 6229 VXD", 0.9, (1, 2, 11, 22))
    assert "/KEY/motion/GRP/MON?data=" in u
    d = json.loads(urllib.parse.unquote(u.split("data=")[1]))
    assert d["matrices"][0]["tag"] == "B 6229 VXD" and d["matrices"][0]["width"] == 10
    # 7) alur penuh dengan detektor dan OCR palsu pada 3 gambar sintetis
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp) / "foto"
        img_dir.mkdir()
        for k in range(3):
            cv2.imwrite(str(img_dir / f"f{k}.jpg"), np.full((200, 300, 3), 80 + k * 40, np.uint8))
        out = Path(tmp) / "out"
        (out / "crops").mkdir(parents=True)
        cfg = argparse.Namespace(gap=1.5, stride=1, margin=0.3, max_frames=0, out=str(out),
                                 perangkat="uji", ocr_width=640, stop_votes=0)
        plates = iter(["H4096GK", "AE6923DO", "HONDA"])
        db = EventDB(str(out / "e.sqlite"))
        with open(out / "k.csv", "w", newline="") as fk:
            lg = csv.writer(fk)
            process(str(img_dir),
                    lambda fr: [(50, 50, 150, 90, 0.9)],
                    lambda im: [(next(plates), 0.9, poly(0, 0, 100, 30))],
                    cfg, db, lg)
        rows = db.con.execute("SELECT plat, suara FROM events ORDER BY id").fetchall()
        assert rows == [("H 4096 GK", 1), ("AE 6923 DO", 1), (None, 0)], rows
        assert len(list((out / "crops").iterdir())) == 3   # termasuk event tak terbaca ("HONDA")

        def run_frames(seq, **kw):
            it = iter(seq)
            frames = ((i, np.full((200, 300, 3), 90, np.uint8), i * 0.1, f"f{i}", False)
                      for i in range(len(seq)))
            c = argparse.Namespace(gap=1.5, stride=1, margin=0.3, max_frames=0, out=str(out),
                                   perangkat="uji", ocr_width=640, stop_votes=kw.get("stop_votes", 0))
            calls = {"n": 0}

            def reader(im):
                calls["n"] += 1
                return next(it)
            with open(out / "kx.csv", "w", newline="") as fk, open(out / "bx.csv", "w", newline="") as fb:
                process("x", lambda fr: [(50, 50, 150, 90, 0.9)], reader, c, db, csv.writer(fk),
                        frames=frames, reads_log=csv.writer(fb))
            row = db.con.execute("SELECT plat, suara, n_baca, n_deteksi, parsial, suara_awalan "
                                 "FROM events ORDER BY id DESC LIMIT 1").fetchone()
            return row, calls["n"], len(open(out / "bx.csv").read().splitlines())

        full = [("H4096GK", 0.9, poly(0, 0, 100, 30))]
        # 8) stop_votes: OCR dilewati setelah 2 suara inti + awalan; satu kendaraan, 8 frame
        row, ncalls, _ = run_frames([full] * 8, stop_votes=2)
        assert ncalls == 2 and row == ("H 4096 GK", 2, 2, 8, None, 2), (row, ncalls)
        # 8b) awalan salah (H) satu kali + inti tanpa awalan: jangan berhenti sebelum awalan benar (N) punya 2 suara
        h_once = [("H3372KM", 0.9, poly(0, 0, 100, 30))]
        no_p = [("3372KM", 0.9, poly(0, 0, 100, 30))]
        n_ok = [("N3372KM", 0.9, poly(0, 0, 100, 30))]
        row, ncalls, _ = run_frames([h_once, no_p, n_ok, n_ok, n_ok, n_ok], stop_votes=2)
        assert row[0] == "N 3372 KM" and ncalls == 4, (row, ncalls)
        # 9) baris masa berlaku saja ("C9-30") di sebagian frame, plat penuh di sisanya
        date_only = [("C9-30", 0.95, poly(0, 40, 60, 18))]
        n5573 = [("5573KC", 0.99, poly(20, 0, 100, 30)), ("N", 0.94, poly(0, 0, 18, 30)),
                 ("C9-30", 0.95, poly(20, 40, 60, 18))]
        row, _, nlog = run_frames([date_only] * 3 + [n5573] * 2)
        assert row[:2] == ("N 5573 KC", 2) and nlog == 5, (row, nlog)  # 5 panggilan OCR tercatat
        # 10) pola nyata dari video: awalan hanya terbaca 1 dari 7 frame, baris masa berlaku
        #     kadang terbaca mirip plat ("C93G") -> tetap "N 5573 KC"
        no_pref = [("5573KC", 1.0, poly(20, 0, 100, 34)), ("C930", 0.94, poly(20, 40, 60, 26))]
        with_pref = [("N5573KC", 0.96, poly(0, 0, 120, 34)), ("C930", 0.94, poly(20, 40, 60, 26))]
        f540 = [("5573KC", 1.0, poly(20, 0, 100, 34)), ("C93G", 0.89, poly(20, 40, 60, 26))]
        row, _, _ = run_frames([no_pref, with_pref, f540, no_pref, no_pref, no_pref, no_pref])
        assert row == ("N 5573 KC", 7, 7, 7, None, 1), row
        # 11) tidak pernah ada awalan -> plat kosong, inti disimpan sebagai parsial
        row, _, _ = run_frames([[("5285KF", 0.99, poly(20, 0, 100, 30))]] * 4)
        assert row[0] is None and row[1] == 4 and row[4] == "5285KF", row
        # 12) hanya angka tanpa huruf seri ("3869") -> tidak ada pembacaan
        row, _, _ = run_frames([[("3869", 1.0, poly(0, 0, 80, 30)), ("0718", 0.94, poly(0, 40, 50, 18))]] * 3)
        assert row[0] is None and row[1] == 0 and row[2] == 0 and row[3] == 3 and row[4] is None, row
        crop_path = db.con.execute("SELECT crop FROM events ORDER BY id DESC LIMIT 1").fetchone()[0]
        assert crop_path and Path(crop_path).exists(), crop_path   # bukti untuk dicek manual
        # 12b) pembaca frame terbaru: pemroses lambat tidak membuat frame menumpuk
        class FakeCap:
            def __init__(self, src):
                self.n = 0

            def read(self):
                time.sleep(0.01)
                self.n += 1
                if self.n > 60:
                    return False, None
                return True, np.full((4, 4, 3), self.n % 250, np.uint8)

            def release(self):
                pass
        rdr = LatestFrameReader("x", cap_factory=FakeCap)
        last, got = 0, []
        while True:
            nxt = rdr.next_frame(last)
            if nxt is None:
                break
            sq, _, tc = nxt
            got.append((sq, (time.time() - tc) * 1000))
            last = sq
            time.sleep(0.1)  # pemroses 10x lebih lambat dari kamera
        rdr.close()
        seqs = [g[0] for g in got]
        assert seqs == sorted(set(seqs)) and len(got) < 25 and seqs[-1] >= 55, (len(got), seqs[-3:])
        assert max(a for _, a in got[:-1]) < 150, [round(a) for _, a in got]  # umur frame tidak menumpuk
        # 12c) sumber langsung: stride diabaikan dan umur/frame dilewati tercatat di log
        live_frames = ((i * 7 + 3, np.full((200, 300, 3), 90, np.uint8), 100 + i * 0.1, f"live{i}", False,
                        {"live": True, "age_ms": 12.5, "dropped": 6}) for i in range(4))
        cfg3 = argparse.Namespace(gap=1.5, stride=5, margin=0.1, max_frames=0, out=str(out),
                                  perangkat="uji", ocr_width=640, stop_votes=0)
        with open(out / "kl.csv", "w", newline="") as fk:
            process("x", lambda fr: [(50, 50, 150, 90, 0.9)],
                    lambda im: [("H4096GK", 0.9, poly(0, 0, 100, 30))],
                    cfg3, db, csv.writer(fk), frames=live_frames)
        rows_l = list(csv.reader(open(out / "kl.csv")))
        assert len(rows_l) == 4 and rows_l[0][-3] == "12.5" and rows_l[0][-2] == "6" and rows_l[0][-1] == "0", rows_l
        # 12d) fragmen awalan + angka: melengkapi inti yang angkanya sama
        assert read_fragment([("N5285", 1.0, 0.99)]) == {"prefix": "N", "digits": "5285"}
        assert read_fragment([("3869", 1.0, 0.9), ("0718", 0.5, 0.9)]) is None   # tanpa awalan: tidak berguna
        assert read_fragment([("5285KF", 1.0, 0.9)]) is None                     # itu inti, bukan fragmen
        fr_n = {"prefix": "N", "digits": "5285"}
        sf = Session(1.0)
        sf.add(0.0, rd("", "5285", "KF"), 0.9, None, (0, 0, 1, 1))
        sf.add(0.1, None, 0.9, None, (0, 0, 1, 1), frag=fr_n)
        f = sf.finalize()
        assert (f["plate"], f["parsial"], f["votes"], f["prefix_votes"], f["frag_votes"]) == \
            ("N 5285 KF", None, 1, 1, 1), f
        # angka fragmen beda (baris masa berlaku "C930") -> diabaikan, tetap parsial
        sf.add(0.0, rd("", "5285", "KF"), 0.9, None, (0, 0, 1, 1))
        sf.add(0.1, None, 0.9, None, (0, 0, 1, 1), frag={"prefix": "C", "digits": "930"})
        f = sf.finalize()
        assert (f["plate"], f["parsial"], f["frag_votes"]) == (None, "5285KF", 0), f
        # fragmen salah baca satu kali (H) kalah dari dua fragmen benar (N)
        sf.add(0.0, rd("", "5285", "KF"), 0.9, None, (0, 0, 1, 1))
        for pf in "HNN":
            sf.add(0.1, None, 0.9, None, (0, 0, 1, 1), frag={"prefix": pf, "digits": "5285"})
        f = sf.finalize()
        assert (f["plate"], f["prefix_votes"], f["frag_votes"]) == ("N 5285 KF", 2, 2), f
        # alur penuh: "N5285" lalu "5285KF" dari dua frame -> satu event lengkap
        row, _, _ = run_frames([[("N5285", 0.99, poly(0, 0, 100, 34))],
                                [("5285KF", 1.0, poly(20, 0, 100, 34)), ("02023", 0.9, poly(20, 40, 60, 20))]])
        assert row == ("N 5285 KF", 1, 1, 2, None, 1), row
        # 12e) OCR di thread terpisah: deteksi tetap jalan saat OCR lambat, hasil tidak hilang
        def run_thread(n_frames, n_vehicle_frames, ocr_s, gap=1.5):
            calls = {"n": 0}

            def slow_read(im):
                calls["n"] += 1
                time.sleep(ocr_s)
                return [("N5285KF", 0.99, poly(0, 0, 120, 34))]
            frames = ((i, np.full((200, 300, 3), 90, np.uint8), i * 0.05, f"t{i}", False)
                      for i in range(n_frames))
            c = argparse.Namespace(gap=gap, stride=1, margin=0.1, max_frames=0, out=str(out),
                                   perangkat="uji", ocr_width=640, stop_votes=0, ocr_mode="thread")
            n0 = db.count()
            with open(out / "kt.csv", "w", newline="") as fk:
                process("x", lambda fr: [(50, 50, 150, 90, 0.9)] if int(fr[0, 0, 0]) == 90 else [],
                        slow_read, c, db, csv.writer(fk), frames=frames)
            evs = db.con.execute("SELECT plat, n_baca, n_deteksi FROM events WHERE id > ? ORDER BY id",
                                 (db.con.execute("SELECT MAX(id) FROM events").fetchone()[0] - (db.count() - n0),)).fetchall()
            lg = list(csv.reader(open(out / "kt.csv")))
            return evs, calls["n"], lg
        evs, ncalls, lg = run_thread(20, 20, 0.15)
        assert evs == [("N 5285 KF", evs[0][1], 20)] and 1 <= evs[0][1] <= 4, evs   # 20 deteksi, hanya beberapa OCR
        assert 1 <= ncalls <= 4 and sum(int(r[-1]) for r in lg) >= 15, (ncalls, [r[-1] for r in lg])
        assert len(lg) == 20 and all(float(r[6]) < 100 for r in lg), [r[6] for r in lg]   # loop utama tidak diblokir OCR
        # 12f) sesi ditutup oleh frame kosong: hasil OCR yang masih berjalan wajib ikut (settle)
        def frames_gap():
            for i in range(6):
                yield i, np.full((200, 300, 3), 90, np.uint8), i * 0.1, f"g{i}", False
            yield 6, np.full((200, 300, 3), 10, np.uint8), 5.0, "g6", False   # tanpa deteksi, 4.5 s kemudian
        c = argparse.Namespace(gap=1.5, stride=1, margin=0.1, max_frames=0, out=str(out),
                               perangkat="uji", ocr_width=640, stop_votes=0, ocr_mode="thread")
        n0 = db.count()
        with open(out / "kg.csv", "w", newline="") as fk:
            process("x", lambda fr: [(50, 50, 150, 90, 0.9)] if int(fr[0, 0, 0]) == 90 else [],
                    lambda im: (time.sleep(0.2), [("N5285KF", 0.99, poly(0, 0, 120, 34))])[1],
                    c, db, csv.writer(fk), frames=frames_gap())
        assert db.count() == n0 + 1
        last = db.con.execute("SELECT plat, n_baca, n_deteksi FROM events ORDER BY id DESC LIMIT 1").fetchone()
        assert last[0] == "N 5285 KF" and last[1] >= 1 and last[2] == 6, last
        # 12g) --cetak-bacaan mencetak bacaan + perkiraan; Ctrl+C menyelesaikan sesi terakhir
        import contextlib, io
        for mode in ("sync", "thread"):
            def frames_ki():
                for i in range(4):
                    yield i, np.full((200, 300, 3), 90, np.uint8), i * 0.1, f"k{i}", False
                raise KeyboardInterrupt
            c = argparse.Namespace(gap=1.5, stride=1, margin=0.1, max_frames=0, out=str(out),
                                   perangkat="uji", ocr_width=640, stop_votes=0, ocr_mode=mode, cetak_bacaan=True)
            n0 = db.count()
            buf = io.StringIO()
            with open(out / "kk.csv", "w", newline="") as fk, contextlib.redirect_stdout(buf):
                process("x", lambda fr: [(50, 50, 150, 90, 0.9)],
                        lambda im: [("N5285KF", 0.99, poly(0, 0, 120, 34))], c, db, csv.writer(fk),
                        frames=frames_ki())
            txt = buf.getvalue()
            assert db.count() == n0 + 1 and "sementara: N 5285 KF" in txt and "Ctrl+C" in txt, (mode, txt)
        # 12h) --tampil: tekan q menghentikan loop dan menyelesaikan sesi; tanpa GUI dimatikan sendiri
        o_show, o_wait = cv2.imshow, cv2.waitKey
        cnt = {"n": 0}

        def fake_show(*a):
            cnt["n"] += 1
        c = argparse.Namespace(gap=1.5, stride=1, margin=0.1, max_frames=0, out=str(out), perangkat="uji",
                               ocr_width=640, stop_votes=0, ocr_mode="sync", tampil=True)

        def frames_t(n=10):
            return ((i, np.full((200, 300, 3), 90, np.uint8), i * 0.1, f"w{i}", False) for i in range(n))
        n0 = db.count()
        try:
            cv2.imshow, cv2.waitKey = fake_show, (lambda ms: ord("q") if cnt["n"] >= 3 else -1)
            with open(out / "kw.csv", "w", newline="") as fk, contextlib.redirect_stdout(io.StringIO()):
                process("x", lambda fr: [(50, 50, 150, 90, 0.9)], lambda im: [("N5285KF", 0.99, poly(0, 0, 120, 34))],
                        c, db, csv.writer(fk), frames=frames_t())
            assert cnt["n"] == 3 and db.count() == n0 + 1, (cnt, db.count() - n0)

            def no_gui(*a):
                raise cv2.error("tidak ada GUI")
            cv2.imshow = no_gui
            buf = io.StringIO()
            with open(out / "kw2.csv", "w", newline="") as fk, contextlib.redirect_stdout(buf):
                process("x", lambda fr: [(50, 50, 150, 90, 0.9)], lambda im: [("N5285KF", 0.99, poly(0, 0, 120, 34))],
                        c, db, csv.writer(fk), frames=frames_t())
            assert "tampilan tidak tersedia" in buf.getvalue() and len(list(csv.reader(open(out / "kw2.csv")))) == 10
        finally:
            cv2.imshow, cv2.waitKey = o_show, o_wait
        # 13) database versi lama dimigrasi
        old = sqlite3.connect(str(out / "lama.sqlite"))
        old.execute("CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, perangkat TEXT, "
                    "sumber TEXT, plat TEXT, suara INTEGER, n_baca INTEGER, n_deteksi INTEGER, "
                    "conf_deteksi REAL, crop TEXT)")
        old.commit()
        old.close()
        db2 = EventDB(str(out / "lama.sqlite"))
        assert {r[1] for r in db2.con.execute("PRAGMA table_info(events)")} >= {"parsial", "suara_awalan", "suara_fragmen"}
        db2.close()
        db.close()
    print("SELFTEST OK")


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="SLPR: YOLOv8 + PaddleOCR + SQLite")
    ap.add_argument("--source", help="folder/foto, video, indeks webcam, atau alamat RTSP")
    ap.add_argument("--model", help="path model YOLO (.onnx atau .pt)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--yolo-device", default="cpu", help="cpu atau 0 (GPU)")
    ap.add_argument("--stride", type=int, default=5, help="proses tiap N frame (video/stream)")
    ap.add_argument("--margin", type=float, default=0.1,
                    help="margin crop plat (relatif). 0.1 dipilih dari uji satu video; 0.3 (versi lama) menurunkan pembacaan awalan, 0.0 sedikit lebih baik tapi berisiko memotong tepi plat")
    ap.add_argument("--gap", type=float, default=1.5, help="detik tanpa deteksi untuk menutup sesi")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--ocr-width", type=int, default=640, help="lebar crop sebelum OCR (kecil = lebih cepat)")
    ap.add_argument("--stop-votes", type=int, default=0,
                    help="berhenti OCR untuk sesi ini bila plat lengkap sudah dapat N suara inti dan N suara awalan "
                         "(0 = selalu OCR). Makin kecil makin hemat, tapi makin besar risiko salah awalan")
    ap.add_argument("--yolo-threads", type=int, default=0,
                    help="batasi thread CPU YOLO (0 = bawaan). Pakai 4 untuk meniru Raspberry Pi 5")
    ap.add_argument("--ocr-threads", type=int, default=0,
                    help="batasi thread CPU PaddleOCR (cpu_threads; 0 = bawaan; belum diverifikasi)")
    ap.add_argument("--ocr-mode", choices=["auto", "sync", "thread"], default="auto",
                    help="sync = OCR di loop utama; thread = OCR di thread pekerja (deteksi tetap jalan); "
                         "auto = thread untuk RTSP/webcam, sync untuk foto/video")
    ap.add_argument("--tampil", action="store_true",
                    help="tampilkan jendela kamera dengan kotak plat dan plat sementara (q atau Esc untuk berhenti)")
    ap.add_argument("--cetak-bacaan", action="store_true",
                    help="cetak tiap hasil OCR dan perkiraan plat sesi saat ini di terminal (berguna untuk kamera langsung)")
    ap.add_argument("--ocr-kw", action="append", default=[],
                    help="argumen tambahan PaddleOCR, mis. --ocr-kw text_det_thresh=0.2 "
                         "--ocr-kw text_det_box_thresh=0.4 (bisa diulang; nama argumen belum diverifikasi)")
    ap.add_argument("--ocr-det-model", default=None, help="nama model deteksi teks PaddleOCR (belum diuji)")
    ap.add_argument("--ocr-rec-model", default=None, help="nama model pembacaan teks PaddleOCR (belum diuji)")
    ap.add_argument("--out", default="hasil_run")
    ap.add_argument("--perangkat", default=platform.node(), help="label perangkat di log")
    ap.add_argument("--shinobi", action="store_true",
                    help="kirim event ke Shinobi (butuh env SHINOBI_URL, SHINOBI_API_KEY, "
                         "SHINOBI_GROUP, SHINOBI_MONITOR)")
    ap.add_argument("--tanpa-thread", action="store_true",
                    help="baca RTSP/webcam berurutan di thread utama (perilaku lama, untuk pembanding); "
                         "frame bisa menumpuk bila pemrosesan lambat")
    ap.add_argument("--selftest", action="store_true", help="uji logika tanpa model")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.source or not args.model:
        ap.error("--source dan --model wajib diisi (atau pakai --selftest)")

    shinobi = None
    if args.shinobi:
        keys = {"base": "SHINOBI_URL", "api_key": "SHINOBI_API_KEY",
                "group": "SHINOBI_GROUP", "monitor": "SHINOBI_MONITOR"}
        shinobi = {k: os.getenv(v) for k, v in keys.items()}
        if not all(shinobi.values()):
            print("peringatan: env Shinobi belum lengkap, event tidak dikirim")
            shinobi = None

    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    detect = make_detector(args.model, args.conf, args.imgsz, args.yolo_device, args.yolo_threads)
    read_text = make_reader(args.ocr_det_model, args.ocr_rec_model, parse_kw(args.ocr_kw), args.ocr_threads)
    db = EventDB(str(out / "events.sqlite"))
    header = ["waktu", "perangkat", "frame", "n_deteksi", "deteksi_ms", "ocr_ms",
              "total_ms", "cpu_persen", "mem_persen", "mem_proses_mb", "suhu_c",
              "umur_frame_ms", "frame_dilewati", "ocr_dilewati"]
    log_path = out / "kinerja.csv"
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            if f.readline().strip().split(",") != header:  # format lama: simpan sebagai arsip
                log_path.rename(out / f"kinerja_lama_{int(time.time())}.csv")
    new_log = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        if new_log:
            log.writerow(header)
        rpath = out / "bacaan.csv"
        rheader = ["waktu", "frame", "plat_pilihan", "ada_awalan", "baris"]  # baris = JSON [[teks, tinggi, skor], ...]
        if rpath.exists():
            with open(rpath, encoding="utf-8") as rf:
                if rf.readline().strip().split(",") != rheader:
                    rpath.rename(out / f"bacaan_lama_{int(time.time())}.csv")
        rnew = not rpath.exists()
        with open(rpath, "a", newline="", encoding="utf-8") as rf:
            reads_log = csv.writer(rf)
            if rnew:
                reads_log.writerow(rheader)
            process(args.source, detect, read_text, args, db, log, shinobi, reads_log=reads_log)
    db.close()
    print("hasil di:", out.resolve())


if __name__ == "__main__":
    main()