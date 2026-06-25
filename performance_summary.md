# Performance Summary: TimesFM 2.5 vs. Chronos-2 vs. Auto ARIMA

This report summarizes the comparative evaluation of **Google's TimesFM 2.5 (200M PyTorch model)**, **Amazon's Chronos-2 (120M foundation model)**, and a traditional statistical model, **Auto ARIMA**, on the 15-minute interval multi-hospital census dataset ([hospital_census.csv](file:///c:/Users/brass/OneDrive/Documents/Projects/google timeseries/hospital_census.csv)).

---

## 📊 Backtesting Evaluation Setup

- **Dataset**: `hospital_census.csv` (1 month of data at 15-minute intervals = 2,880 total steps per series)
- **Series Evaluated**: 7 medsurg units across 2 hospitals (`H1_Medsurg_A`, `H1_Medsurg_B`, `H1_Medsurg_C`, `H2_Medsurg_A`, `H2_Medsurg_B`, `H2_Medsurg_C`, `H2_Medsurg_D`)
- **Context Length**: 256 steps (64 hours of historical context)
- **Forecast Horizon**: 64 steps (16 hours ahead)
- **Backtesting Framework**: 4 sliding windows, rolling forward by 96 steps (24 hours) per window. 
- **Total Evaluations**: 28 distinct time-series forecast slices (7 series × 4 windows)

---

## 📈 Accuracy and Speed Metrics

The table below shows the overall average metrics calculated across all 7 series and all 4 evaluation windows:

| Evaluation Metric | TimesFM 2.5 | Chronos-2 | Auto ARIMA | TimesFM Gain vs. ARIMA | Chronos-2 Gain vs. ARIMA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MAE** (Mean Absolute Error) | **1.2154** | **1.0469** | 1.5347 | **20.8%** | **31.8%** |
| **RMSE** (Root Mean Squared Error) | **1.4506** | **1.2554** | 1.8339 | **20.9%** | **31.5%** |
| **MAPE** (Mean Absolute Percentage Error) | **8.99%** | **7.61%** | 11.54% | **22.1%** | **34.1%** |
| **Average Inference Time** (per window) | **~1.07s** (batch) | **~0.09s** (batch) | **~8.35s** (sequential) | **7.8x faster** | **92.8x faster** |

---

## 💡 Key Performance Observations

### 1. Superior Accuracy of Foundation Models
Both foundation models consistently outperformed Auto ARIMA. **Chronos-2** yielded the lowest overall forecasting error, achieving a **~31.8% lower MAE compared to Auto ARIMA**, and a **~13.9% lower MAE compared to TimesFM 2.5**. TimesFM 2.5 itself achieved a **~20.8% lower MAE compared to Auto ARIMA**. The zero-shot capabilities of both pre-trained foundation models allow them to capture complex hospital census trends without any local training or parameter fitting.

### 2. High Computational Throughput (Batching & GPU Acceleration)
* **Chronos-2** is extremely efficient, utilizing a non-autoregressive encoder structure to predict all quantiles in a single forward pass. Under GPU acceleration, it processed all 7 medsurg series in a single parallel operation in **~0.09 seconds total** (~0.01s per series), making it **92.8x faster** than Auto ARIMA.
* **TimesFM 2.5** also leverages batch forecasting, processing all 7 series in **~1.07 seconds total** (~0.15s per series) on the GPU, which is **7.8x faster** than Auto ARIMA.
* **Auto ARIMA** required sequential fitting and parameter search for each series individually, taking **~8.35 seconds** for the 7 series per window.
* **Scaling Advantage**: As the workload scales to hundreds of units/facilities, Auto ARIMA's CPU runtime scales linearly, creating a significant bottleneck. In contrast, both foundation models support batch inference and GPU acceleration to run in near-constant time.

---

## 🔍 Visual Comparison Results

Below is the visualization of the rolling forecasts for the representative unit (`H1_Medsurg_A`) alongside the aggregate metrics comparison:

![Hospital Census Backtest Comparison Plot](timesfm_vs_arima_backtest.png)

*The plot displays the last 50 historical steps in black (for context), the actual target values in blue, TimesFM 2.5 forecasts in red (dashed), Chronos-2 forecasts in purple (dash-dotted), and Auto ARIMA forecasts in green (dotted).*

---

## 📈 AutoGluon Ensemble vs. Raw Chronos-2 (Forward Testing)

To explore ensembling strategies, we evaluated an **AutoGluon Timeseries Ensemble** (combining Chronos-Bolt, Seasonal Naive, and Theta models) against raw **Chronos-2** on the rolling 24-hour day-ahead forecast over the 5-day hold-out hospital dataset:

| Evaluation Metric | Raw Chronos-2 | AutoGluon Ensemble | Ensemble Gain vs. Chronos-2 |
| :--- | :---: | :---: | :---: |
| **MAE** | 1.1258 | **1.0772** | **+4.3%** |
| **RMSE** | 1.3370 | **1.2889** | **+3.6%** |
| **MAPE** | 7.70% | **7.40%** | **+3.9%** |

*Note: Ensembling local models (like Theta and Seasonal Naive) with deep learning representation models (Chronos-Bolt) successfully reduced overall error metrics by **3.6% to 4.3%** compared to a raw zero-shot foundation model.*

---

## 🚀 Recommendations for Hospital Forecasting

1. **Deploy AutoGluon Ensembles**: Combining pre-trained foundation models with simple local baselines yields the highest forecast accuracy, providing robust guardrails against individual model failures.
2. **Adopt Foundation Models**: If ensembling pipelines are not yet established, zero-shot models like **Chronos-2** and **TimesFM 2.5** are highly recommended over traditional sequential statistical models due to their superior zero-shot accuracy and computational speed.
3. **MLOps Simplification**: Pretrained foundation models negate the need for periodic retraining on local data, significantly reducing infrastructure and computation overhead.
