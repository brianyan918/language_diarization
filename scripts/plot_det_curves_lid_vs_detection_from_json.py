import json
import matplotlib.pyplot as plt
import numpy as np

# Paths to your DET data files
lid_path = "model/spoken-language-diarization/exp/runs/inf_langdiar_whisper_multi/model_ablation.cls.xtts.whispersmall.on.v3_csfl_read/det_curve_det_data.json"
det_path = "model/spoken-language-diarization/exp/runs/inf_langdiar_whisper_multi/model_ablation.detection.xtts.whispersmall.on.v3_csfl_read/det_curve_det_data.json"

# Load data
with open(lid_path, "r") as f:
    lid_data = json.load(f)
with open(det_path, "r") as f:
    det_data = json.load(f)

# Extract values
lid_fa = [d["fa_rate"] for d in lid_data]
lid_miss = [d["miss_rate"] for d in lid_data]
lid_thr = [d["threshold"] for d in lid_data]

det_fa = [d["fa_rate"] for d in det_data]
det_miss = [d["miss_rate"] for d in det_data]
det_thr = [d["threshold"] for d in det_data]

plt.figure(figsize=(8, 6))
plt.plot(lid_fa, lid_miss, 'o-', label='LID', color='blue')
plt.plot(det_fa, det_miss, 's--', label='Detection', color='red')

for i, t in enumerate(lid_thr):
    plt.annotate(f"{t:.2f}", (lid_fa[i], lid_miss[i]), textcoords="offset points", xytext=(5,5), fontsize=8, color='blue')
for i, t in enumerate(det_thr):
    plt.annotate(f"{t:.2f}", (det_fa[i], det_miss[i]), textcoords="offset points", xytext=(5,-10), fontsize=8, color='red')

plt.xlabel("False Alarm Rate (FA)")
plt.ylabel("Miss Rate (Miss)")
plt.title("DET Curves: LID vs Detection")
plt.xlim(-0.05, 0.7)
plt.ylim(-0.05, 0.4)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("det_curves_lid_vs_detection_from_json.png", dpi=150)
print("Plot saved as det_curves_lid_vs_detection_from_json.png")
