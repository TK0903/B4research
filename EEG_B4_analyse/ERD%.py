import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# =====================
# 設定（あなたのXDFに合わせて修正済み）
# =====================
XDF_PATH = r"E:\EEG_B4\sub-P002\ses-S002\eeg\sub-P002_ses-S002_task-MI_run-003_eeg.xdf"

# どっちでもOK：名前指定（今回のXDFに一致）
EEG_STREAM_NAME = "OpenBCI_Cyton_EEG"
MARKER_STREAM_NAME = "MI_Markers"

CHS = ["C3", "C4"]
ALPHA_BAND = (8, 13)

TMIN, TMAX = -0.5, 2.0
BASELINE = (-1.0, 0.0)

# ERD強度の評価窓（cue開始後）
SCORE_WIN = (0.0, 2.0)

# =====================
# util
# =====================
def bandpass(x, fs, fmin, fmax, order=4):
    b, a = butter(order, [fmin/(fs/2), fmax/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def erd_percent_from_power(power_ts, times, baseline):
    b0, b1 = baseline
    bmask = (times >= b0) & (times <= b1)
    bmean = power_ts[bmask].mean()
    return (power_ts - bmean) / bmean * 100.0

def trial_score(erd_c3, erd_c4, times, win):
    w0, w1 = win
    wmask = (times >= w0) & (times <= w1)
    # ERDは負方向なので「どれだけ下がったか」を正の値にする
    s3 = (-erd_c3[wmask]).max()
    s4 = (-erd_c4[wmask]).max()
    return max(s3, s4)

def normalize_ch(s: str) -> str:
    return s.strip().upper().replace(".", "").replace(" ", "")

# =====================
# XDF load
# =====================
streams, _ = pyxdf.load_xdf(XDF_PATH)

# --- 名前で拾う（あなたのXDFに一致） ---
eeg_stream = next(s for s in streams if s["info"]["name"][0] == EEG_STREAM_NAME)
marker_stream = next(s for s in streams if s["info"]["name"][0] == MARKER_STREAM_NAME)

# （より安全にするなら下の2行に置き換えてもOK）
# eeg_stream = next(s for s in streams if s["info"]["type"][0].lower() == "eeg")
# marker_stream = next(s for s in streams if s["info"]["type"][0].lower() == "markers")

# EEG
eeg = eeg_stream["time_series"].T  # (n_ch, n_samp)
fs = float(eeg_stream["info"]["nominal_srate"][0])
eeg_t0 = float(eeg_stream["time_stamps"][0])

# チャンネル名取得
ch_meta = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
ch_names = [normalize_ch(ch_meta[i]["label"][0]) for i in range(len(ch_meta))]

idx_c3 = ch_names.index("C3")
idx_c4 = ch_names.index("C4")

# Marker
markers = [str(x[0]).lower() for x in marker_stream["time_series"]]
marker_times = marker_stream["time_stamps"]

# =====================
# α power (Hilbert)
# =====================
alpha_c3 = np.abs(hilbert(bandpass(eeg[idx_c3], fs, *ALPHA_BAND)))**2
alpha_c4 = np.abs(hilbert(bandpass(eeg[idx_c4], fs, *ALPHA_BAND)))**2

# epoch time axis
n_samp = int(round((TMAX - TMIN) * fs))
times = np.arange(n_samp) / fs + TMIN

# =====================
# trialごとにERD計算して保存
# =====================
trials = {"left": [], "right": []}

for k, (lab, t_evt) in enumerate(zip(markers, marker_times)):
    if lab not in ["left", "right"]:
        continue

    center = int(round((t_evt - eeg_t0) * fs))
    start = center + int(round(TMIN * fs))
    end = start + n_samp

    if start < 0 or end > len(alpha_c3):
        continue

    p3 = alpha_c3[start:end]
    p4 = alpha_c4[start:end]

    erd3 = erd_percent_from_power(p3, times, BASELINE)
    erd4 = erd_percent_from_power(p4, times, BASELINE)

    score = trial_score(erd3, erd4, times, SCORE_WIN)

    trials[lab].append({
        "trial_index": k,
        "event_time": float(t_evt),
        "erd_c3": erd3,
        "erd_c4": erd4,
        "score": float(score),
    })

# =====================
# 左右それぞれ best trial を選んで表示
# =====================
for cond in ["left", "right"]:
    if len(trials[cond]) == 0:
        print(f"No trials for {cond}")
        continue

    best = max(trials[cond], key=lambda d: d["score"])
    print(f"[{cond}] best trial index={best['trial_index']}  score={best['score']:.2f}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(times, best["erd_c4"], label="C4")
    ax.axvline(0, linestyle="--", color="k")
    ax.axhline(0, linestyle="--", color="k")
    ax.set_xlim(TMIN, TMAX)
    ax.set_ylim(-120, 60)
    ax.set_xlabel("Time (s) (cue onset = 0)")
    ax.set_ylabel("ERD (%) (baseline ratio)")
    ax.set_title("Left MI trial")
    ax.legend()

    ax.tick_params(direction="in", top=True, right=True)

    plt.tight_layout()
    plt.show()
