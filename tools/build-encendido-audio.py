"""
Arma assets/encendido.mp3: el megamix de G.E.M. sincronizado con "El encendido".

Toma un fragmento de cada canción (el arranque de su línea más repetida, o sea el coro,
alineado al beat) y los encadena con cruces suaves en los mismos tiempos que la animación.

Uso (con el venv del Listening Room, que trae numpy + soundfile con MP3):
  python tools/build-encendido-audio.py "<carpeta con los WAV>" "<carpeta analysis/*.json>"

Los tiempos de abajo DEBEN coincidir con los de index.html (SL, DURS, GEM, SWITCH, REVEAL, END).
"""
import base64, collections, glob, json, math, os, sys
import numpy as np
import soundfile as sf

WAV_DIR, ANALYSIS_DIR = sys.argv[1], sys.argv[2]
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "encendido.mp3")

# ---- tiempos (ms), espejo de index.html ----
jsround = lambda v: math.floor(v + 0.5)
N = 13
SL = 2400
DURS = [jsround(2480 - i * 340 / (N - 1)) for i in range(N)]
STARTS = [SL + sum(DURS[:i]) for i in range(N)]
GEM = STARTS[-1] + DURS[-1] + 200
SWITCH = GEM + 2900
REVEAL = SWITCH + 400
END = REVEAL + 1750
XA = 260          # medio cruce de audio (ms) alrededor de cada cambio
FADE_IN = 1600    # la música entra mientras florece la luz
FADE_OUT = 2200   # y se va mientras se abre la página

SR = 48000
# si un fragmento no convence, fija aquí el segundo de arranque a mano: {número de pista: segundo}
OVERRIDES = {}


def ms(v):
    return int(round(v / 1000 * SR))


def anchor_for(analysis):
    """Segundo de la canción donde arranca el fragmento: inicio de la línea más repetida (2a vez si existe), al beat."""
    lines = analysis.get("lyrics", {}).get("lines", [])
    beats = analysis["events"]["beats"]["items"]
    dur = analysis["audio"]["duration"]
    mix = np.frombuffer(base64.b64decode(analysis["curves"]["channels"]["mix"]), dtype=np.uint8).astype(float)
    rate = analysis["curves"]["rate"]

    def energy(t, span=2.5):
        a, b = int(t * rate), int((t + span) * rate)
        return mix[a:b].mean() if b > a else 0

    anchor = None
    if lines:
        count = collections.Counter(l["text"].strip().lower() for l in lines)
        top, n = count.most_common(1)[0]
        hits = [l["start"] for l in lines if l["text"].strip().lower() == top]
        if n >= 2:
            anchor = hits[1] if len(hits) > 1 else hits[0]
    # respaldo / validación: debe ser una zona con energía (≥ percentil 70 de la pista)
    p70 = np.percentile(mix[mix > 0], 70) if (mix > 0).any() else 0
    if anchor is None or energy(anchor) < p70:
        cands = np.arange(dur * 0.2, dur * 0.8, 0.5)
        anchor = float(max(cands, key=energy))
    # al beat más cercano (antes o igual)
    prev = [b for b in beats if b <= anchor + 0.05]
    return prev[-1] if prev else anchor


def load(path):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    assert sr == SR, f"{path}: {sr} Hz"
    return data[:, :2] if data.shape[1] >= 2 else np.repeat(data, 2, axis=1)


wavs = sorted(glob.glob(os.path.join(WAV_DIR, "[0-9][0-9]. *.wav")))
jsons = sorted(glob.glob(os.path.join(ANALYSIS_DIR, "[0-9][0-9]-*.json")))
assert len(wavs) == N and len(jsons) == N, (len(wavs), len(jsons))

total = ms(END)
mixbuf = np.zeros((total, 2), dtype=np.float32)
segs = []
for i, (w, j) in enumerate(zip(wavs, jsons)):
    analysis = json.load(open(j, encoding="utf-8"))
    song = load(w)
    anc = OVERRIDES.get(i + 1) or anchor_for(analysis)
    # ventana del fragmento en la línea de tiempo de la animación
    a = 0 if i == 0 else STARTS[i] - XA
    b = END if i == N - 1 else STARTS[i] + DURS[i] + XA
    src0 = anc + (a - STARTS[i]) / 1000           # segundo de la canción que suena en t = a
    src0 = max(0.0, src0)
    n = ms(b) - ms(a)
    s0 = int(src0 * SR)
    seg = song[s0:s0 + n]
    if len(seg) < n:
        seg = np.pad(seg, ((0, n - len(seg)), (0, 0)))
    # envolvente: cruces de potencia constante
    env = np.ones(n, dtype=np.float32)
    fin = ms(FADE_IN) if i == 0 else ms(2 * XA)
    fout = ms(FADE_OUT) if i == N - 1 else ms(2 * XA)
    x = np.linspace(0, 1, fin, dtype=np.float32); env[:fin] *= np.sin(x * np.pi / 2)
    x = np.linspace(0, 1, fout, dtype=np.float32); env[-fout:] *= np.cos(x * np.pi / 2)
    rms = float(np.sqrt(np.mean(seg[fin:n - fout] ** 2))) or 1e-6
    segs.append((i, a, seg * env[:, None], rms))
    print(f"{i+1:02d} {os.path.basename(w)[4:-4]:<24} coro~{anc:6.2f}s  fragmento {src0:6.2f}s +{n/SR:4.2f}s  rms {20*math.log10(rms):5.1f} dB")

# nivela: todos al RMS mediano (máx. +6 dB de ganancia)
target = float(np.median([s[3] for s in segs]))
for i, a, seg, rms in segs:
    g = min(target / rms, 2.0)
    s = ms(a)
    mixbuf[s:s + len(seg)] += seg * g

peak = float(np.abs(mixbuf).max())
if peak > 0.97:
    mixbuf *= 0.97 / peak
sf.write(OUT, mixbuf, SR, format="MP3", subtype="MPEG_LAYER_III")
print(f"-> {os.path.normpath(OUT)}  {total/SR:.1f}s  pico {20*math.log10(max(peak,1e-6)):.1f} dBFS  {os.path.getsize(OUT)//1024} KB")
