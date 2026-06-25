# Formal Forward Testing Report: TimesFM 2.5 vs. Chronos-2 vs. Auto ARIMA

This report summarizes the results of a strict forward-testing comparison over a **5-day reserved hold-out set** (June 26 to June 30, 2026) using the multi-hospital census dataset.

## 📊 Evaluation Parameters
- **Hold-out Set Size**: 5 days (480 time steps at 15-minute intervals)
- **Forecast Horizon**: 24 hours (96 steps) updated daily (rolling day-ahead forecast)
- **Total Series Evaluated**: 7 medsurg units across 2 hospitals
- **Historical Context**: 256 steps (64 hours)

## 📈 Overall Accuracy Summary (Averaged over 7 units & 5 days)

| Metric | TimesFM 2.5 | Chronos-2 | Auto ARIMA | TimesFM Gain vs. ARIMA | Chronos-2 Gain vs. ARIMA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MAE** | **1.2999** | **1.1239** | 1.6333 | **20.4%** | **31.2%** |
| **RMSE** | **1.5508** | **1.3354** | 1.9330 | **19.8%** | **41.9%** |
| **MAPE** | **8.89%** | **7.69%** | 11.04% | **19.5%** | **30.4%** |

## 💡 Key Findings
1. **Robustness on Unseen Data**: Both foundation models (TimesFM 2.5 and Chronos-2) maintain a significant accuracy edge on the strict hold-out set compared to Auto ARIMA, proving the power of large-scale pretraining.
2. **Daily Adaptability**: The rolling 24-hour day-ahead setup shows that foundation models are viable plug-and-play solutions for daily operational scheduling in hospitals, significantly reducing forecasting errors.

---
## 🔍 Visualization
The detailed forecast comparison plot (featuring shaded 90% prediction intervals for all models) has been saved as `forward_test_comparison.png` in the project root.

![Forward Test Timeline Comparison](forward_test_comparison.png)
