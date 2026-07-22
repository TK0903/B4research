import mne
import matplotlib.pyplot as plt
import numpy as np

edf_path = r"e:\EEG_Dataset\eeg-motor-movementimagery-dataset-1.0.0\S001\S001R01.edf"
target_ch = "C3"
show_seconds = 10

# EDF読み込み（MNEは物理値[Volt]を返す）
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
sfreq = raw.info["sfreq"]

# 実際に保存されていた整数レンジを推定する
# （MNEの内部変換は linear_map: dig→phy, つまり phy = dig * slope + offset）
dig_min, dig_max = raw._raw_extras[0]["dig_min"], raw._raw_extras[0]["dig_max"]
phy_min, phy_max = raw._raw_extras[0]["phys_min"], raw._raw_extras[0]["phys_max"]

# スケーリングを逆算： dig ≈ (phy - offset)/slope
slope = (phy_max - phy_min) / (dig_max - dig_min)
offset = phy_min - slope * dig_min

# データ取得
if target_ch not in raw.ch_names:
    for ch in raw.ch_names:
        if target_ch.lower() in ch.lower():
            target_ch = ch
            break
data, times = raw.get_data(picks=[target_ch], return_times=True)
data = data[0]
N = int(show_seconds * sfreq)
data = data[:N]
times = times[:N]

# MNE出力[V] → EDF定義の物理値単位
phy_vals = data
# 物理値を生整数に“逆変換”
dig_vals = (phy_vals - offset) / slope

plt.figure(figsize=(10, 3))
plt.plot(times, dig_vals)
plt.xlabel("Time [s]")
plt.ylabel(f"{target_ch} (digital counts ≈)")
plt.title(f"{target_ch} (first {show_seconds}s, approximate raw counts)")
plt.tight_layout()
plt.show()
