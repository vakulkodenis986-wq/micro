import json, os
import numpy as np
import sounddevice as sd
from scipy.signal import sosfilt, butter
import tkinter as tk
from tkinter import ttk
import pystray
from PIL import Image, ImageDraw

SR, BLOCK = 48000, 480  # 48 кГц, блоки по 10 мс
CFG = os.path.join(os.getenv("APPDATA", "."), "MicStudio.json")
DEF = dict(mic="", out="", gate=-50.0, warmth=2.0, presence=3.0, air=2.0,
           thr=-20.0, ratio=3.0, gain=3.0, bypass=False)


def biquad(kind, f0, db):
    """Биквад-фильтр (RBJ): peak / low / high shelf -> строка SOS."""
    A = 10 ** (db / 40)
    w = 2 * np.pi * f0 / SR
    c, s = np.cos(w), np.sin(w)
    if kind == "peak":
        al = s / 2
        b = [1 + al * A, -2 * c, 1 - al * A]
        a = [1 + al / A, -2 * c, 1 - al / A]
    else:
        sq = 2 * np.sqrt(A) * (s / 2 * np.sqrt(2))
        if kind == "low":
            b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c),
                 A * ((A + 1) - (A - 1) * c - sq)]
            a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c),
                 (A + 1) + (A - 1) * c - sq]
        else:
            b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c),
                 A * ((A + 1) + (A - 1) * c - sq)]
            a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c),
                 (A + 1) - (A - 1) * c - sq]
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1, a[1] / a[0], a[2] / a[0]]


class Engine:
    def __init__(s, p):
        s.p, s.stream = p, None
        s.zi, s.gg, s.gr, s.level = np.zeros((4, 2)), 0.0, 0.0, -90.0
        s.rebuild()

    def rebuild(s):
        p = s.p
        s.sos = np.vstack([butter(2, 80, "hp", fs=SR, output="sos"),
                           biquad("low", 200, p["warmth"]),
                           biquad("peak", 3500, p["presence"]),
                           biquad("high", 10000, p["air"])])

    def cb(s, indata, outdata, frames, t, status):
        p = s.p
        x = indata[:, 0].astype(np.float64)
        y, s.zi = sosfilt(s.sos, x, zi=s.zi)
        lvl = 20 * np.log10(np.sqrt(np.mean(y * y)) + 1e-9)
        s.level = lvl
        if p["bypass"]:
            y = x
        else:
            n = len(y)
            # шумоподавитель (гейт)
            tgt = 1.0 if lvl > p["gate"] else 0.0
            g0 = s.gg
            s.gg += (tgt - s.gg) * (0.6 if tgt > s.gg else 0.04)
            y = y * np.linspace(g0, s.gg, n)
            # компрессор + громкость
            tgr = -max(0.0, lvl - p["thr"]) * (1 - 1 / p["ratio"])
            c0 = s.gr
            s.gr += (tgr - s.gr) * (0.4 if tgr < s.gr else 0.05)
            gain = 10 ** ((np.linspace(c0, s.gr, n) + p["gain"]) / 20)
            y = 0.9 * np.tanh(y * gain / 0.9)  # мягкий лимитер
        outdata[:, 0] = y
        outdata[:, 1] = y

    def start(s, i_in, i_out):
        s.stop()
        s.rebuild()
        ws = sd.WasapiSettings(auto_convert=True)
        s.stream = sd.Stream(samplerate=SR, blocksize=BLOCK, device=(i_in, i_out),
                             channels=(1, 2), dtype="float32", latency="low",
                             callback=s.cb, extra_settings=(ws, ws))
        s.stream.start()

    def stop(s):
        if s.stream:
            s.stream.close()
            s.stream = None


def devices():
    h = next(i for i, x in enumerate(sd.query_hostapis()) if "WASAPI" in x["name"])
    ds = [(i, d) for i, d in enumerate(sd.query_devices()) if d["hostapi"] == h]
    ins = {d["name"]: i for i, d in ds if d["max_input_channels"] > 0}
    outs = {d["name"]: i for i, d in ds if d["max_output_channels"] > 0}
    return ins, outs


def make_icon():
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((22, 6, 42, 38), 10, fill=(60, 180, 120))
    d.arc((14, 20, 50, 50), 0, 180, fill=(60, 180, 120), width=4)
    d.line((32, 50, 32, 60), fill=(60, 180, 120), width=4)
    return im


class App:
    def __init__(s):
        s.p = dict(DEF)
        try:
            s.p.update(json.load(open(CFG, encoding="utf-8")))
        except Exception:
            pass
        s.eng = Engine(s.p)
        s.ins, s.outs = devices()
        r = s.root = tk.Tk()
        r.title("Mic Studio")
        r.geometry("360x640")
        r.protocol("WM_DELETE_WINDOW", r.withdraw)  # крестик = в трей

        def combo(text, vals, key, auto=""):
            ttk.Label(r, text=text).pack(anchor="w", padx=10, pady=(8, 0))
            v = tk.StringVar(value=s.p[key] if s.p[key] in vals else
                             next((n for n in vals if auto and auto in n), ""))
            cb = ttk.Combobox(r, textvariable=v, values=list(vals), state="readonly")
            cb.pack(fill="x", padx=10)
            cb.bind("<<ComboboxSelected>>", lambda e: s.restart())
            return v

        s.mic = combo("Микрофон (вход):", s.ins, "mic")
        s.out = combo("Выход (виртуальный микрофон):", s.outs, "out", "CABLE Input")

        def slider(label, key, lo, hi, res=0.5):
            sc = tk.Scale(r, from_=lo, to=hi, resolution=res, orient="horizontal",
                          label=label, command=lambda v: s.set(key, float(v)))
            sc.set(s.p[key])
            sc.pack(fill="x", padx=10)

        slider("Шумоподавитель: порог, дБ", "gate", -70, -20)
        slider("Тепло (низкие), дБ", "warmth", -6, 8)
        slider("Разборчивость (3.5 кГц), дБ", "presence", -4, 8)
        slider("Воздух (высокие), дБ", "air", -4, 8)
        slider("Компрессор: порог, дБ", "thr", -40, -5)
        slider("Компрессор: степень", "ratio", 1, 8)
        slider("Громкость на выходе, дБ", "gain", -6, 12)

        s.bp = tk.BooleanVar(value=s.p["bypass"])
        ttk.Checkbutton(r, text="Обход (без обработки)", variable=s.bp,
                        command=lambda: s.set("bypass", s.bp.get())).pack(pady=4)
        s.meter = ttk.Progressbar(r, maximum=100)
        s.meter.pack(fill="x", padx=10, pady=4)
        s.status = tk.StringVar(value="Остановлено")
        ttk.Label(r, textvariable=s.status).pack()
        s.btn = ttk.Button(r, text="Старт", command=s.toggle)
        s.btn.pack(pady=6)

        s.tray = pystray.Icon("MicStudio", make_icon(), "Mic Studio", pystray.Menu(
            pystray.MenuItem("Открыть", lambda: r.after(0, r.deiconify), default=True),
            pystray.MenuItem("Старт/Стоп", lambda: r.after(0, s.toggle)),
            pystray.MenuItem("Выход", lambda: r.after(0, s.quit))))
        s.tray.run_detached()
        r.after(100, s.toggle)
        s.tick()

    def set(s, k, v):
        s.p[k] = v
        s.eng.rebuild()

    def toggle(s):
        if s.eng.stream:
            s.eng.stop()
            s.status.set("Остановлено")
            s.btn.config(text="Старт")
            return
        try:
            s.eng.start(s.ins[s.mic.get()], s.outs[s.out.get()])
            s.status.set("Работает")
            s.btn.config(text="Стоп")
        except Exception as e:
            s.status.set(f"Ошибка: {str(e)[:60]}")

    def restart(s):
        if s.eng.stream:
            s.toggle()
        s.toggle()

    def tick(s):
        s.meter["value"] = max(0, (s.eng.level + 80) * 100 / 80) if s.eng.stream else 0
        s.root.after(50, s.tick)

    def quit(s):
        s.p["mic"], s.p["out"] = s.mic.get(), s.out.get()
        json.dump(s.p, open(CFG, "w", encoding="utf-8"))
        s.eng.stop()
        s.tray.stop()
        s.root.destroy()


if __name__ == "__main__":
    App().root.mainloop()
