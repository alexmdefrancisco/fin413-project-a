# FIN-413 Project A: Risk‐Based and Momentum‐Enhanced Portfolio Construction

This directory contains the **final code** for Project A of FIN-413. It performs:

1. **Part 1**: Exploratory Data Analysis & log-returns preparation  
2. **Part 2**: Covariance estimation, cleaning, and dependence analysis  
3. **Part 3**: Risk-based portfolios (MV, ERC, ENB, HRP)  
4. **Part 4**: HRP extensions with alternative distance metrics (HRPe) and a momentum-enhanced HRPe (TSM-HRPe)  

> **Note**: You will also find Jupyter notebooks in the `notebooks/` folder. These are **preliminary** exploratory versions created during development and are **not** invoked by `main.py`. The canonical implementation lives in `src/*.py`.

---

## 📂 Code Structure
```text
code/
├── notebooks/         # Preliminary EDA & prototyping (not part of final pipeline)
├── src/
│   ├── part1.py       # EDA & data prep
│   ├── part2.py       # Covariance estimation & analysis
│   ├── part3.py       # Risk‐based portfolios
│   ├── part4.py       # HRPe & TSM-HRPe
│   ├── utils.py       # I/O helpers (e.g. load_cov)
│   └── hrp_helpers.py # Shared HRP routines & copula distances
├── main.py            # Entry point
├── config.yaml        # Paths, dates, windows parameters
├── requirements.txt   # Python dependencies
└── README.md          
```

---

## Installation

```bash
# (Optional) create & activate venv
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
.\.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the pipeline via `main.py`:

```bash
# All parts in sequence
python main.py --step all

# Or individual steps:
python main.py --step part1      # EDA & prepare log-returns
python main.py --step part2      # Covariance estimation & analysis
python main.py --step part3      # Risk‐based portfolios
python main.py --step part4      # HRPe & TSM-HRPe
```