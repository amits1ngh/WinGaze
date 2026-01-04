**WinGaze** is a **rerun and PyQt5 based application** for aligning multimodal signals, csv export, analysis, and visualization in Human Robot Interaction experiments.  
It integrates timeline annotations with synchronized video playback, performs real time MediaPipe-based hand tracking, and provides real-time plots and video panels in Rerun.

---

## 🚀 Features
✅ Load **ELAN timeline annotation files (.txt)** to segment videos  
✅ Synchronize a **main video** with optional **eye-tracking metrics/video**  
✅ Run **hand tracking** on frames using **MediaPipe Hands**  
✅ Display:
- Velocity over time
- Mean X and Y hand positions
✅ Export all tracking data to **CSV**  
✅ Filter by action hand **Left / Right / Both hands**  
✅ PyQt5 GUI with:
- Dropdowns for timeline selection
- Rerun viewer for video panels and time-series plotting

---

## 🎬 Screenshots
<div align="center">
  <img src="uploads/WinGaze-demo.png" width="600" alt="WinGaze GUI">
</div>

---

## 🔧 Installation
📌 **Python 3.9+ recommended**

1️⃣ Clone this repository:
```bash
git clone https://github.com/yourusername/WinGaze.git
cd WinGaze
pip install -r requirements.txt
```

2️⃣ Run the application:
```bash
python WinGaze.py
```

The Rerun viewer will open in a separate window to display the video panels and hand-metric time series.

---

## Project Structure
```
WinGaze/
  src/
    config/    # App and rerun settings
    core/      # Hand tracking and core data types
    data_io/   # ELAN readers and CSV export
    ui/        # PyQt5 window and controls
    utils/     # Qt helper utilities
    vis/       # Rerun logging and layouts
  WinGaze.py   # Entry point wrapper
```

---

## Publication
If you use WinGaze in academic work, please cite:

Singh, A., Wrede, B., Birte, R., Groß, A., & Rohlfing, K. J. (2025). "Manners Matter: Action history guides attention and repair choices during interaction." *IEEE International Conference on Development and Learning*. https://doi.org/10.1109/ICDL63968.2025.11204385
