# Formal Forward Testing Report: TimesFM 2.5 vs. Auto ARIMA

This report summarizes the results of a strict forward-testing comparison over a **5-day reserved hold-out set** (June 26 to June 30, 2026) using the multi-hospital census dataset.

## 📊 Evaluation Parameters
- **Hold-out Set Size**: 5 days (480 time steps at 15-minute intervals)
- **Forecast Horizon**: 24 hours (96 steps) updated daily (rolling day-ahead forecast)
- **Total Series Evaluated**: 7 medsurg units across 2 hospitals
- **Historical Context**: 256 steps (64 hours)

## 📈 Overall Accuracy Summary (Averaged over 7 units & 5 days)

| Metric | TimesFM 2.5 | Auto ARIMA | Performance Gain (TimesFM) |
| :--- | :---: | :---: | :---: |
| **MAE** | **1.2999** | 1.6333 | **20.4% error reduction** |
| **RMSE** | **1.5508** | 1.9330 | **19.8% error reduction** |
| **MAPE** | **8.89%** | 11.04% | **19.5% error reduction** |

## 💡 Key Findings
1. **Robustness on Unseen Data**: TimesFM 2.5 maintains its accuracy edge on the strict hold-out set, proving its capability to generalize well without target overfitting.
2. **Daily Adaptability**: The rolling 24-hour day-ahead setup shows that TimesFM is a viable plug-and-play solution for daily operational scheduling in hospitals, significantly reducing forecasting errors compared to classical statistical baselines.

---
## 🔍 Visualization
The detailed forecast comparison plot has been saved as `forward_test_comparison.png` in the project root.
