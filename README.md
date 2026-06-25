# Hospital Census Forecasting: Google TimesFM 2.5 vs. Amazon Chronos-2 vs. Auto ARIMA

This repository contains a performance comparison between **Google's TimesFM 2.5 (200M parameter PyTorch Foundation Model)**, **Amazon's Chronos-2 (120M parameter PyTorch Foundation Model)**, and a traditional **Auto ARIMA** statistical model. All models are evaluated on a simulated 15-minute interval multi-hospital census dataset.

---

## 📊 Overview & Methodology

Hospital census forecasting is crucial for daily staffing, resource allocation, and operational planning. This project compares three contrasting methodologies:
1. **TimesFM 2.5**: A zero-shot foundation model pre-trained on massive time-series corpuses. It requires **no training/fitting** on local data and supports batch inference.
2. **Chronos-2**: A zero-shot foundation model that outputs quantiles directly in a single forward pass without autoregressive sampling, natively supporting multivariate and covariate forecasting.
3. **Auto ARIMA**: A classical statistical model from `pmdarima` that fits parameters ($p, d, q$) locally and sequentially for each individual time series.

---

## 📈 Performance Summary (Strict Hold-Out Set)

The models were evaluated using a rolling 24-hour day-ahead forecast (96 steps) over a strict **5-day hold-out dataset** (June 26 to June 30, 2026) across 7 unit series.

### 1. Accuracy (Averaged over 7 units and 5 days)

| Metric | Google TimesFM 2.5 | Amazon Chronos-2 | Auto ARIMA | TimesFM Gain vs. ARIMA | Chronos-2 Gain vs. ARIMA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MAE** (Mean Absolute Error) | **1.2999** | **1.1237** | 1.6333 | **20.4%** | **31.2%** |
| **RMSE** (Root Mean Squared Error) | **1.5508** | **1.3350** | 1.9330 | **19.8%** | **30.9%** |
| **MAPE** (Mean Absolute Percentage Error) | **8.89%** | **7.69%** | 11.04% | **19.5%** | **30.3%** |

### 2. Computational Profile & Speeds (Averaged over 5 days under GPU Acceleration)

| Profile Area | Google TimesFM 2.5 | Amazon Chronos-2 | Auto ARIMA | Comparison Details |
| :--- | :---: | :---: | :---: | :--- |
| **Training / Fitting** | **None** (Zero-Shot) | **None** (Zero-Shot) | Fits parameters per window | Foundation models do not require local training, eliminating MLOps retraining pipelines. |
| **Inference Mode** | **Batch Processing** | **Batch Processing** | Sequential Processing | Both foundation models process all 7 series in parallel. ARIMA must run sequentially. |
| **Time per Window (7 series)** | **~0.98s** total (~0.14s/series) | **~0.07s** total (~0.01s/series) | **~7.03s** total (~1.00s/series) | **Chronos-2 is ~100x faster** and **TimesFM is ~7x faster** than Auto ARIMA. |

---

## 🔍 Visual Results

### 1. Sliding-Window Backtesting (4 Windows)
The plot below displays the predictions of TimesFM 2.5, Chronos-2, and Auto ARIMA across the sliding evaluation windows:

![Hospital Census Backtest Comparison Plot](timesfm_vs_arima_backtest.png)

*The plot displays the last 50 historical steps in black (for context), the actual target values in blue, TimesFM 2.5 forecasts in red (dashed), Chronos-2 forecasts in purple (dash-dotted), and Auto ARIMA forecasts in green (dotted).*

### 2. Stitched 5-Day Forward Test Comparison
The plot below illustrates the daily rolling forecasts stitched together over the 5-day hold-out period for the representative unit `H1_Medsurg_A`:

![Forward Test Timeline Comparison](forward_test_comparison.png)

*The top panel shows the actual census (blue) vs. TimesFM 2.5 (red dashed), Chronos-2 (purple dash-dotted), and Auto ARIMA (green dotted). Vertical line indicators show the rolling daily forecast origins. Bottom panels show aggregate errors.*

### 3. Hospital Census Profiles
The month-long 15-minute interval dataset exhibits clear daily admissions/discharges cycles and weekly staffing trends:

![Hospital Census Profile Plot](hospital_census_plot.png)

---

## 📁 Repository Structure

* `generate_hospital_data.py`: Creates `hospital_census.csv` with simulated 15-minute census values (bounded between 10 and 20) showing daily/weekly hospital admission behaviors.
* `plot_hospital_data.py`: Computes hospital census totals and saves the 3-panel profile plot (`hospital_census_plot.png`).
* `backtest_comparison.py`: Runs walk-forward sliding window backtesting over the development split.
* `forward_test.py`: Partitions dataset and runs the formal forward-testing daily forecast loop over the hold-out set, generating `forward_test_comparison.png` and `forward_test_summary.md`.
* `test_timesfm.py`: Simple smoke test script to verify TimesFM package and downloads weights from Hugging Face.
* `requirements.txt`: Package dependencies (`timesfm`, `chronos-forecasting`, `accelerate`, `pmdarima`, `pandas`, `matplotlib`).

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.10+ (compatible up to Python 3.14). Create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Dataset & Plot Profiles
```powershell
python generate_hospital_data.py
python plot_hospital_data.py
```

### 3. Run Sliding Window Backtesting
```powershell
python backtest_comparison.py
```

### 4. Run Strict Hold-Out Forward Testing
```powershell
python forward_test.py
```
