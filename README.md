<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=250&section=header&text=Log%20AI%20Inspector&fontSize=60&animation=fadeIn&fontAlignY=38&desc=GRU%20%C2%B7%20Soft%20Attention%20%C2%B7%20Real-Time%20Log%20Anomaly%20Detection&descAlignY=55&descAlign=50" width="100%" />
</div>

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://log-ai-inspector-final-year-project.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

<div align="center">
  <h3>⚡ A premium AI-powered dashboard for detecting anomalies in system logs<br/>using a GRU encoder with soft attention — no labelled data required.</h3>
</div>

---

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Star-Struck.png" alt="Star-Struck" width="25" height="25" /> Project Highlights

* <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png" alt="Robot" width="20" height="20" /> **GRU-Based Anomaly Detection**: A Gated Recurrent Unit (GRU) encoder processes log sequences and predicts an anomaly probability score from 0 to 1 for each window of log lines.
* <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/High%20Voltage.png" alt="High Voltage" width="20" height="20" /> **Soft Attention — Explainability Built In**: The model computes per-line attention weights, pinpointing *exactly* which log lines contributed most to an anomaly flag — no black-box guessing.
* <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Bar%20Chart.png" alt="Bar Chart" width="20" height="20" /> **Interactive Plotly Dashboard**: Real-time anomaly timeline, attention heatmap across all sequences, and a root-cause explorer with highlighted log lines — all rendered with Plotly.
* <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Magnifying%20Glass%20Tilted%20Right.png" alt="Magnifying Glass" width="20" height="20" /> **Zero Training Required**: A deterministic hash-based tokenizer converts raw log text to integer sequences — works on *any* log format without a pre-trained vocabulary or labelled dataset.
* <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="20" height="20" /> **One-Click Cloud Deploy**: Runs natively on Streamlit Cloud — upload a log file or load the bundled demo with a single button click.

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Film%20Frames.png" alt="Demo" width="25" height="25" /> Live Demo

> 🌐 **[https://log-ai-inspector-final-year-project.streamlit.app/](https://log-ai-inspector-final-year-project.streamlit.app/)**

Click **🚀 Load Demo Logs** on the app to instantly analyse the bundled sample log file — no upload needed.

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Gear.png" alt="Gear" width="25" height="25" /> How It Works

```
Raw Log File (.txt)
       │
       ▼
┌─────────────────────────────┐
│  utils.py — Preprocessing   │
│  • Strip timestamps & IPs   │
│  • Hash-encode each line    │
│  • Slide window (size=10)   │
└────────────┬────────────────┘
             │  sequences: List[List[int]]
             ▼
┌─────────────────────────────┐
│  model.py — GRU + Attention │
│  • Embedding layer          │
│  • GRU encoder              │
│  • Soft-attention pooling   │
│  • FC → Sigmoid → score     │
└────────────┬────────────────┘
             │  score [0,1]  +  attention weights
             ▼
┌─────────────────────────────┐
│  app.py — Streamlit UI      │
│  • KPI metric cards         │
│  • Anomaly timeline chart   │
│  • Root-cause log viewer    │
│  • Full attention heatmap   │
└─────────────────────────────┘
```

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/File%20Folder.png" alt="File Folder" width="25" height="25" /> Project Structure

```
Log-AI-Inspector/
│
├── app.py              # Streamlit dashboard — UI, charts, and inference orchestration
├── model.py            # GRU + soft-attention anomaly detection model (PyTorch)
├── utils.py            # Log preprocessing: cleaning, tokenising, windowing
├── sample_logs.txt     # Bundled sample log file for demo mode
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Dark theme + headless server config for Streamlit Cloud
└── .gitignore
```

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Scroll.png" alt="Scroll" width="25" height="25" /> Workflow

| Step | Module | Description |
| :---: | :--- | :--- |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Magnifying%20Glass%20Tilted%20Right.png" width="28" /> **1. Ingest** | `app.py` | User uploads a `.txt` / `.log` file or loads the bundled demo via sidebar. |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Soap.png" width="28" /> **2. Clean** | `utils.py` | Timestamps, IPv4 addresses, block IDs, and plain numbers are normalised or removed to surface semantic tokens. |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Symbols/Triangular%20Ruler.png" width="28" /> **3. Tokenise** | `utils.py` | Each cleaned line is hashed deterministically into an integer token in `[1, VOCAB_SIZE-1]` — no vocabulary file needed. |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Puzzle%20Piece.png" width="28" /> **4. Window** | `utils.py` | Lines are grouped into sliding windows of 10. Short final windows are zero-padded so every sequence is the same length. |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png" width="28" /> **5. Inference** | `model.py` | Each sequence passes through the GRU encoder. The soft-attention head produces a context vector; the FC head outputs an anomaly score `[0, 1]`. |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Activities/Trophy.png" width="28" /> **6. Visualise** | `app.py` | Scores and attention weights are rendered as an anomaly timeline, a per-sequence log viewer with highlighted lines, and a full attention heatmap. |

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="30" height="30" /> Quick Start

### <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Clipboard.png" alt="Clipboard" width="25" height="25" /> Prerequisites

* [Python 3.9+](https://python.org/)
* [Git](https://git-scm.com/)
* `pip` (bundled with Python)

---

### <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Hammer%20and%20Wrench.png" alt="Setup" width="25" height="25" /> Local Installation — Step by Step

<img align="right" src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Laptop.png" width="120" height="120" alt="Laptop"/>

**1. Clone the repository**
```bash
git clone https://github.com/SubashSK777/Log-AI-Inspector.git
cd Log-AI-Inspector
```

**2. Create and activate a virtual environment** *(recommended)*
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit app**
```bash
streamlit run app.py
```

**5. Open in browser**

Streamlit will print a local URL — typically:
```
Local URL: http://localhost:8501
```
Navigate there and click **🚀 Load Demo Logs** to see the app in action instantly.

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Bar%20Chart.png" alt="Bar Chart" width="28" height="28" /> Dashboard Features

| Feature | Description |
| :--- | :--- |
| **KPI Metric Cards** | At-a-glance counts of total sequences, critical anomalies, warnings, healthy windows, and peak score. |
| **Anomaly Timeline** | Plotly line chart of all anomaly scores with colour-coded markers (🔴 critical / 🟡 warning / ⚪ healthy), configurable threshold bands, and optional moving average overlay. |
| **Root Cause Explorer** | Slider-driven sequence inspector showing per-line attention weights as a bar chart and a scrollable log viewer with attention-highlighted lines. |
| **Attention Heatmap** | Full matrix heatmap (sequences × log lines) revealing which lines the model consistently focuses on across the entire file. |
| **Configurable Thresholds** | Sidebar sliders for anomaly threshold, warning threshold, and attention highlight sensitivity — all update in real time. |

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png" alt="Robot" width="28" height="28" /> Model Architecture

```
Input: LongTensor (batch=1, seq_len=10)
         │
         ▼
   Embedding(vocab=500, dim=64)
         │
         ▼
   GRU(input=64, hidden=64, batch_first=True)
         │ outputs: (1, 10, 64)
         ▼
   Linear(64 → 1)  ──▶  Softmax(dim=1)  =  Attention Weights (1, 10, 1)
         │
   Weighted Sum  →  Context Vector (1, 64)
         │
         ▼
   Linear(64 → 1)  ──▶  Sigmoid  =  Anomaly Score [0, 1]
```

| Component | Details |
| :--- | :--- |
| **Embedding** | `vocab_size=500`, `embed_dim=64`, `padding_idx=0` |
| **Encoder** | 1-layer GRU, `hidden_dim=64`, `batch_first=True` |
| **Attention** | Additive soft-attention, softmax-normalised over sequence dimension |
| **Head** | Single FC layer + Sigmoid → scalar anomaly probability |
| **Tokeniser** | Polynomial rolling hash (base 31, mod `VOCAB_SIZE-1`) — deterministic, no training |

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Hammer.png" alt="Tech" width="25" height="25" /> Tech Stack

| Layer | Technology |
| :--- | :--- |
| **UI / Dashboard** | [Streamlit](https://streamlit.io/) ≥ 1.32 |
| **Deep Learning** | [PyTorch](https://pytorch.org/) ≥ 2.2 (CPU) |
| **Visualisation** | [Plotly](https://plotly.com/python/) ≥ 5.19 |
| **Data Processing** | [Pandas](https://pandas.pydata.org/) ≥ 2.1, [NumPy](https://numpy.org/) ≥ 1.26 |
| **Hosting** | [Streamlit Community Cloud](https://streamlit.io/cloud) |

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Globe%20with%20Meridians.png" alt="Deploy" width="25" height="25" /> Deploy to Streamlit Cloud

1. Fork or push this repo to your GitHub account.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → **New app**.
3. Fill in:
   - **Repository**: `SubashSK777/Log-AI-Inspector`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy** — done in ~2 minutes! 🎉

> The `.streamlit/config.toml` in this repo automatically applies the dark theme and `headless` server mode required for cloud deployment.

<br/>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Handshakes/Handshake.png" alt="Contributing" width="28" height="28" /> Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "feat: add amazing feature"`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

<br/>

---

<div align="center">
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="50" height="50" />
  <h3>Built with ❤️ as a Final Year Project</h3>
  <p>
    <a href="https://log-ai-inspector-final-year-project.streamlit.app/">🌐 Live Demo</a> •
    <a href="https://github.com/SubashSK777/Log-AI-Inspector/issues">🐛 Report Bug</a> •
    <a href="https://github.com/SubashSK777/Log-AI-Inspector/issues">✨ Request Feature</a>
  </p>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>
</div>
