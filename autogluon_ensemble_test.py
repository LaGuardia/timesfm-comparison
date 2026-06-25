import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from chronos import Chronos2Pipeline
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

def calculate_metrics(y_true, y_pred):
    """
    Computes standard forecasting error metrics: MAE, RMSE, and MAPE.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # Prevent division by zero
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def main():
    print("=== AutoGluon Chronos Ensemble vs. Raw Chronos-2 Forward Testing ===")
    
    csv_file = "hospital_census.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run generate_hospital_data.py first.")
        return

    # Load data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded dataset: {df.shape}")
    
    # Identify target columns
    series_cols = [col for col in df.columns if col != 'timestamp']
    num_series = len(series_cols)
    
    # Partition configurations
    dev_len = 2400        # Split point index (first 25 days)
    context_len = 256     # Historical context length passed to models
    horizon_len = 96      # 24-hour ahead lookahead window
    num_days = 5          # Hold-out days
    
    # 1. Fit AutoGluon Predictor on Development Set
    print("\nPreparing development data for AutoGluon...")
    dev_df = df.iloc[:dev_len]
    dev_df_long = dev_df.melt(id_vars=["timestamp"], var_name="item_id", value_name="target")
    dev_data = TimeSeriesDataFrame.from_data_frame(
        dev_df_long,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    
    print("\nInitializing AutoGluon TimeSeriesPredictor...")
    predictor = TimeSeriesPredictor(
        prediction_length=horizon_len,
        target="target",
        eval_metric="MAE",
        quantile_levels=[0.1, 0.5, 0.9]
    )
    
    print("Fitting AutoGluon Ensemble model on development data...")
    predictor.fit(
        dev_data,
        hyperparameters={
            "Chronos": {"model_path": "bolt_base"},
            "SeasonalNaive": {},
            "Theta": {}
        },
        time_limit=180
    )
    print("AutoGluon Ensemble fit successfully.")
    
    # 2. Load Raw Chronos-2 Model
    print("\nLoading Raw Chronos-2 model from Hugging Face...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chronos_pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    print("Raw Chronos-2 model loaded successfully.")
    
    # Choose H1_Medsurg_A for visual timeline stitching
    viz_col = "H1_Medsurg_A"
    viz_idx = series_cols.index(viz_col)
    
    # Lists to store stitched results for visual timeline plotting
    stitched_actual = []
    stitched_chronos = []
    stitched_ensemble = []
    
    stitched_chronos_lower = []
    stitched_chronos_upper = []
    stitched_ensemble_lower = []
    stitched_ensemble_upper = []
    
    # Lists to store daily error metrics
    chronos_metrics_all = []
    ensemble_metrics_all = []
    
    # Loop day-by-day through hold-out period
    for d in range(num_days):
        origin = dev_len + d * horizon_len
        history_start = origin - context_len
        history_end = origin
        target_start = origin
        target_end = origin + horizon_len
        
        print(f"\n--- Forward Test Day {d+1}/{num_days} ---")
        print(f"History range: [{history_start}:{history_end}] | Target range: [{target_start}:{target_end}]")
        
        history_list = [df[col].values[history_start:history_end] for col in series_cols]
        actual_list = [df[col].values[target_start:target_end] for col in series_cols]
        
        # A. Raw Chronos-2 Forecast
        t0 = time.time()
        with torch.no_grad():
            chronos_forecast = chronos_pipeline.predict(
                inputs=history_list,
                prediction_length=horizon_len
            )
        chronos_preds = []
        chronos_lower_90 = []
        chronos_upper_90 = []
        for f in chronos_forecast:
            f_np = f.cpu().numpy()
            f_np = np.squeeze(f_np, axis=0) # shape (21, horizon)
            chronos_preds.append(f_np[10, :])     # Median (quantile 0.5)
            chronos_lower_90.append(f_np[1, :])   # P5
            chronos_upper_90.append(f_np[19, :])  # P95
        chronos_preds = np.array(chronos_preds)
        chronos_lower_90 = np.array(chronos_lower_90)
        chronos_upper_90 = np.array(chronos_upper_90)
        chronos_time = time.time() - t0
        
        # B. AutoGluon Ensemble Forecast
        t0 = time.time()
        # Prepare history context for AutoGluon
        history_df = df.iloc[history_start:history_end]
        history_df_long = history_df.melt(id_vars=["timestamp"], var_name="item_id", value_name="target")
        history_data = TimeSeriesDataFrame.from_data_frame(
            history_df_long,
            id_column="item_id",
            timestamp_column="timestamp"
        )
        # Generate predictions
        ensemble_forecast = predictor.predict(history_data)
        
        ensemble_preds = []
        ensemble_lower_90 = []
        ensemble_upper_90 = []
        for col in series_cols:
            col_preds = ensemble_forecast.xs(col, level="item_id")
            mean_vals = col_preds["mean"].values
            p10_vals = col_preds["0.1"].values
            p90_vals = col_preds["0.9"].values
            # Extrapolate 80% interval (P10 to P90) to 90% interval using Normal scaling
            half_width_90 = (p90_vals - p10_vals) * 0.6417
            lower_90 = mean_vals - half_width_90
            upper_90 = mean_vals + half_width_90
            
            ensemble_preds.append(mean_vals)
            ensemble_lower_90.append(lower_90)
            ensemble_upper_90.append(upper_90)
        ensemble_preds = np.array(ensemble_preds)
        ensemble_lower_90 = np.array(ensemble_lower_90)
        ensemble_upper_90 = np.array(ensemble_upper_90)
        ensemble_time = time.time() - t0
        
        print(f"Raw Chronos-2 inference time: {chronos_time:.2f}s")
        print(f"AutoGluon Ensemble inference time: {ensemble_time:.2f}s")
        
        # Compute and record daily metrics
        chronos_day_metrics = []
        ensemble_day_metrics = []
        for i, col in enumerate(series_cols):
            chronos_m = calculate_metrics(actual_list[i], chronos_preds[i])
            ens_m = calculate_metrics(actual_list[i], ensemble_preds[i])
            
            chronos_day_metrics.append(chronos_m)
            ensemble_day_metrics.append(ens_m)
            
            chronos_metrics_all.append(chronos_m)
            ensemble_metrics_all.append(ens_m)
            
        chronos_day_avg = pd.DataFrame(chronos_day_metrics).mean()
        ens_day_avg = pd.DataFrame(ensemble_day_metrics).mean()
        print(f"Day {d+1} Average Metrics (Across {num_series} units):")
        print(f"  Raw Chronos-2      -> MAE: {chronos_day_avg['MAE']:.4f} | RMSE: {chronos_day_avg['RMSE']:.4f} | MAPE: {chronos_day_avg['MAPE']:.2f}%")
        print(f"  AutoGluon Ensemble -> MAE: {ens_day_avg['MAE']:.4f} | RMSE: {ens_day_avg['RMSE']:.4f} | MAPE: {ens_day_avg['MAPE']:.2f}%")
        
        # Collect values for visual stitching
        stitched_actual.extend(actual_list[viz_idx])
        stitched_chronos.extend(chronos_preds[viz_idx])
        stitched_ensemble.extend(ensemble_preds[viz_idx])
        
        stitched_chronos_lower.extend(chronos_lower_90[viz_idx])
        stitched_chronos_upper.extend(chronos_upper_90[viz_idx])
        stitched_ensemble_lower.extend(ensemble_lower_90[viz_idx])
        stitched_ensemble_upper.extend(ensemble_upper_90[viz_idx])
        
    # Calculate overall metrics across all windows and series
    chronos_df = pd.DataFrame(chronos_metrics_all)
    ens_df = pd.DataFrame(ensemble_metrics_all)
    
    chronos_avg = chronos_df.mean()
    ens_avg = ens_df.mean()
    
    print("\n" + "="*50)
    print("=== FINAL COMPARISON RESULTS OVER HOLD-OUT SET ===")
    print(f"Raw Chronos-2      -> MAE: {chronos_avg['MAE']:.4f} | RMSE: {chronos_avg['RMSE']:.4f} | MAPE: {chronos_avg['MAPE']:.2f}%")
    print(f"AutoGluon Ensemble -> MAE: {ens_avg['MAE']:.4f} | RMSE: {ens_avg['RMSE']:.4f} | MAPE: {ens_avg['MAPE']:.2f}%")
    print("="*50)
    
    # Save the report to autogluon_ensemble_summary.md in workspace
    summary_md_path = "autogluon_ensemble_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"""# Forward Testing Report: Raw Chronos-2 vs. AutoGluon Ensemble
        
This report evaluates the performance of a raw pretrained **Chronos-2** foundation model against an **AutoGluon Timeseries Ensemble** (combining Chronos-Bolt, Seasonal Naive, and Theta models) on the 5-day rolling hospital census hold-out test set.

## 📊 Evaluation Parameters
- **Hold-out Set Size**: 5 days (480 steps at 15-minute intervals)
- **Forecast Horizon**: 24 hours (96 steps) updated daily (rolling day-ahead forecast)
- **Total Series Evaluated**: 7 medsurg units across 2 hospitals
- **Historical Context**: 256 steps (64 hours)

## 📈 Overall Accuracy Summary (Averaged over 7 units & 5 days)

| Metric | Raw Chronos-2 | AutoGluon Ensemble | Ensemble Gain vs. Chronos-2 |
| :--- | :---: | :---: | :---: |
| **MAE** | **{chronos_avg['MAE']:.4f}** | **{ens_avg['MAE']:.4f}** | **{((chronos_avg['MAE'] - ens_avg['MAE'])/chronos_avg['MAE'] * 100):.1f}%** |
| **RMSE** | **{chronos_avg['RMSE']:.4f}** | **{ens_avg['RMSE']:.4f}** | **{((chronos_avg['RMSE'] - ens_avg['RMSE'])/chronos_avg['RMSE'] * 100):.1f}%** |
| **MAPE** | **{chronos_avg['MAPE']:.2f}%** | **{ens_avg['MAPE']:.2f}%** | **{((chronos_avg['MAPE'] - ens_avg['MAPE'])/chronos_avg['MAPE'] * 100):.1f}%** |

## 💡 Key Findings
1. **Ensemble Performance**: The AutoGluon Ensemble successfully combines the strength of pretrained Chronos models with robust local baselines (Seasonal Naive, Theta), yielding balanced and competitive results.
2. **Computational Trade-offs**: While raw Chronos-2 is a highly powerful foundation model, the AutoGluon Ensemble provides a structured pipelines approach that integrates multiple models to mitigate individual model failures.

---
## 🔍 Visualization
The detailed forecast comparison plot (featuring shaded 90% prediction intervals for both models) has been saved as `autogluon_ensemble_comparison.png` in the project root.

![Autogluon Ensemble Timeline Comparison](autogluon_ensemble_comparison.png)
""")
    print(f"\nSummary report saved successfully to: {os.path.abspath(summary_md_path)}")
    
    # Plotting
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Stitched Forecast Timeline Subplot (occupies top half)
    ax_timeline = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    timestamps = df['timestamp'].iloc[dev_len : dev_len + len(stitched_actual)].values
    
    ax_timeline.plot(timestamps, stitched_actual, label="Actual Census", color="blue", linewidth=2.0, zorder=3)
    ax_timeline.plot(timestamps, stitched_chronos, label="Raw Chronos-2 Forecast", color="purple", linestyle="-.", linewidth=1.5, zorder=2)
    ax_timeline.plot(timestamps, stitched_ensemble, label="AutoGluon Ensemble Forecast", color="orange", linestyle="--", linewidth=1.5, zorder=2)
    
    # Shade 90% prediction intervals
    ax_timeline.fill_between(timestamps, stitched_chronos_lower, stitched_chronos_upper, color="purple", alpha=0.10, label="Raw Chronos-2 90% PI", zorder=1)
    ax_timeline.fill_between(timestamps, stitched_ensemble_lower, stitched_ensemble_upper, color="orange", alpha=0.10, label="AutoGluon Ensemble 90% PI", zorder=1)
    
    # Draw vertical lines for day transitions
    for d in range(1, num_days):
        origin = dev_len + d * horizon_len
        day_boundary_ts = df['timestamp'].iloc[origin]
        ax_timeline.axvline(x=day_boundary_ts, color="gray", linestyle="-.", alpha=0.7)
        ax_timeline.text(day_boundary_ts, max(stitched_actual) - 1, f" Day {d+1} Origin", color="gray", fontsize=9)
        
    ax_timeline.set_title(f"Stitched 5-Day Rolling Forecast Timeline ({viz_col})", fontsize=14, fontweight="bold")
    ax_timeline.set_xlabel("Date & Time")
    ax_timeline.set_ylabel("Census Count")
    ax_timeline.legend(loc="upper left")
    
    # Format x-axis dates nicely
    import matplotlib.dates as mdates
    ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax_timeline.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate() # Auto-rotate date labels
    
    ax_timeline.grid(True, linestyle="--", alpha=0.5)
    
    # 2. MAE & RMSE Bar Chart (bottom left)
    ax_bar1 = plt.subplot2grid((2, 2), (1, 0))
    metrics_to_plot = ["MAE", "RMSE"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    chronos_vals = [chronos_avg["MAE"], chronos_avg["RMSE"]]
    ens_vals = [ens_avg["MAE"], ens_avg["RMSE"]]
    
    ax_bar1.bar(x - width/2, chronos_vals, width, label="Raw Chronos-2", color="purple", alpha=0.8)
    ax_bar1.bar(x + width/2, ens_vals, width, label="AutoGluon Ensemble", color="orange", alpha=0.8)
    ax_bar1.set_title("Average MAE & RMSE (Hold-Out)", fontsize=12, fontweight="bold")
    ax_bar1.set_xticks(x)
    ax_bar1.set_xticklabels(metrics_to_plot)
    ax_bar1.set_ylabel("Error Value")
    ax_bar1.legend()
    ax_bar1.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    # 3. MAPE Bar Chart (bottom right)
    ax_bar2 = plt.subplot2grid((2, 2), (1, 1))
    ax_bar2.bar([0], [chronos_avg["MAPE"]], width, label="Raw Chronos-2", color="purple", alpha=0.8)
    ax_bar2.bar([1], [ens_avg["MAPE"]], width, label="AutoGluon Ensemble", color="orange", alpha=0.8)
    ax_bar2.set_title("Average MAPE % (Hold-Out)", fontsize=12, fontweight="bold")
    ax_bar2.set_xticks([0, 1])
    ax_bar2.set_xticklabels(["Raw Chronos-2", "AutoGluon Ensemble"])
    ax_bar2.set_ylabel("Percentage (%)")
    ax_bar2.legend()
    ax_bar2.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_filename = "autogluon_ensemble_comparison.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"Comparison plot saved successfully to: {os.path.abspath(plot_filename)}")
    print("AutoGluon Ensemble comparison workflow completed successfully!")

if __name__ == "__main__":
    main()
