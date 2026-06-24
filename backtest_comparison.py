import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import timesfm

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
    print("=== TimesFM 2.5 vs. Auto ARIMA Backtesting on Hospital Census ===")
    
    csv_file = "hospital_census.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run generate_hospital_data.py first.")
        return

    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded hospital census dataset with shape: {df.shape}")
    
    # Identify the series columns (exclude timestamp)
    series_cols = [col for col in df.columns if col != 'timestamp']
    print(f"Evaluating {len(series_cols)} series: {', '.join(series_cols)}")
    
    # Backtest configuration
    context_len = 256     # 64 hours history (4 steps/hour)
    horizon_len = 64      # 16 hours forecast
    step_size = 96        # 24 hours roll size
    num_windows = 4       # Number of rolling windows
    
    # Load TimesFM Model
    print("\nLoading TimesFM model from Hugging Face...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=horizon_len,
            normalize_inputs=True,
        )
    )
    print("Model compiled successfully.")
    
    # Results storage
    timesfm_all_metrics = []
    arima_all_metrics = []
    
    # Representative unit to visualize: H1_Medsurg_A
    viz_col = "H1_Medsurg_A"
    viz_col_idx = series_cols.index(viz_col)
    
    # Create matplotlib subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for k in range(num_windows):
        # Calculate indices to slide the window backward from the end
        test_start = len(df) - horizon_len - (num_windows - 1 - k) * step_size
        test_end = test_start + horizon_len
        history_start = test_start - context_len
        history_end = test_start
        
        print(f"\n--- Running Window {k+1}/{num_windows} ---")
        print(f"History Steps: [{history_start}:{history_end}] | Test Steps: [{test_start}:{test_end}]")
        
        # Prepare inputs for batch forecasting
        history_list = [df[col].values[history_start:history_end] for col in series_cols]
        actual_targets = [df[col].values[test_start:test_end] for col in series_cols]
        
        # 1. TimesFM Batch Forecast (predicts all 7 units in one call!)
        t0 = time.time()
        forecast_results = model.forecast(
            horizon=horizon_len,
            inputs=history_list
        )
        timesfm_preds = forecast_results[0] # Shape: (num_series, horizon)
        timesfm_time = time.time() - t0
        
        # 2. Auto ARIMA Sequential Forecast (fits and predicts units one by one)
        t0 = time.time()
        arima_preds = []
        for history_data in history_list:
            arima_model = pm.auto_arima(
                history_data,
                seasonal=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            pred = arima_model.predict(n_periods=horizon_len)
            arima_preds.append(pred)
        arima_preds = np.array(arima_preds) # Shape: (num_series, horizon)
        arima_time = time.time() - t0
        
        print(f"TimesFM (Batch) Inference Time: {timesfm_time:.2f}s")
        print(f"Auto ARIMA (Sequential) Fit/Predict Time: {arima_time:.2f}s")
        
        # Calculate and record metrics for each series in this window
        tfm_window_metrics = []
        arima_window_metrics = []
        for i, col in enumerate(series_cols):
            tfm_m = calculate_metrics(actual_targets[i], timesfm_preds[i])
            ari_m = calculate_metrics(actual_targets[i], arima_preds[i])
            
            tfm_window_metrics.append(tfm_m)
            arima_window_metrics.append(ari_m)
            
            # Save for overall average
            timesfm_all_metrics.append(tfm_m)
            arima_all_metrics.append(ari_m)
            
        # Compute window average across all units
        tfm_win_avg = pd.DataFrame(tfm_window_metrics).mean()
        ari_win_avg = pd.DataFrame(arima_window_metrics).mean()
        print(f"Window {k+1} Average Metrics (Across {len(series_cols)} units):")
        print(f"  TimesFM   -> MAE: {tfm_win_avg['MAE']:.4f} | RMSE: {tfm_win_avg['RMSE']:.4f} | MAPE: {tfm_win_avg['MAPE']:.2f}%")
        print(f"  AutoARIMA -> MAE: {ari_win_avg['MAE']:.4f} | RMSE: {ari_win_avg['RMSE']:.4f} | MAPE: {ari_win_avg['MAPE']:.2f}%")
        
        # Plot predictions for the representative unit (H1_Medsurg_A)
        ax = axes[k]
        history_plot_len = 50
        history_indices = np.arange(history_end - history_plot_len, history_end)
        ax.plot(history_indices, history_list[viz_col_idx][-history_plot_len:], label="History (Partial)", color="black", alpha=0.6)
        
        test_indices = np.arange(test_start, test_end)
        ax.plot(test_indices, actual_targets[viz_col_idx], label="Actual Target", color="blue", linewidth=1.5)
        ax.plot(test_indices, timesfm_preds[viz_col_idx], label="TimesFM 2.5", color="red", linestyle="--")
        ax.plot(test_indices, arima_preds[viz_col_idx], label="Auto ARIMA", color="green", linestyle=":")
        
        ax.set_title(f"Window {k+1} Forecast ({viz_col})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Census Count")
        ax.legend()
        ax.grid(True)
        
    # Calculate overall average metrics across all windows and series
    tfm_df = pd.DataFrame(timesfm_all_metrics)
    ari_df = pd.DataFrame(arima_all_metrics)
    
    tfm_avg = tfm_df.mean()
    ari_avg = ari_df.mean()
    
    print("\n=== Overall Average Performance (7 units x 4 windows) ===")
    print(f"TimesFM 2.5   -> MAE: {tfm_avg['MAE']:.4f} | RMSE: {tfm_avg['RMSE']:.4f} | MAPE: {tfm_avg['MAPE']:.2f}%")
    print(f"Auto ARIMA    -> MAE: {ari_avg['MAE']:.4f} | RMSE: {ari_avg['RMSE']:.4f} | MAPE: {ari_avg['MAPE']:.2f}%")
    
    # Plot overall MAE & RMSE comparison
    metrics_to_plot = ["MAE", "RMSE"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    ax_metrics = axes[4]
    tfm_vals = [tfm_avg["MAE"], tfm_avg["RMSE"]]
    ari_vals = [ari_avg["MAE"], ari_avg["RMSE"]]
    
    ax_metrics.bar(x - width/2, tfm_vals, width, label="TimesFM 2.5", color="red", alpha=0.8)
    ax_metrics.bar(x + width/2, ari_vals, width, label="Auto ARIMA", color="green", alpha=0.8)
    
    ax_metrics.set_title("Overall MAE & RMSE (Lower is Better)")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metrics_to_plot)
    ax_metrics.set_ylabel("Error Value")
    ax_metrics.legend()
    ax_metrics.grid(True, axis='y')
    
    # Plot overall MAPE comparison
    ax_mape = axes[5]
    ax_mape.bar([0], [tfm_avg["MAPE"]], width, label="TimesFM 2.5", color="red", alpha=0.8)
    ax_mape.bar([1], [ari_avg["MAPE"]], width, label="Auto ARIMA", color="green", alpha=0.8)
    
    ax_mape.set_title("Overall MAPE % (Lower is Better)")
    ax_mape.set_xticks([0, 1])
    ax_mape.set_xticklabels(["TimesFM 2.5", "Auto ARIMA"])
    ax_mape.set_ylabel("Percentage (%)")
    ax_mape.legend()
    ax_mape.grid(True, axis='y')
    
    # Adjust layout and save plot
    plt.tight_layout()
    output_plot = "timesfm_vs_arima_backtest.png"
    plt.savefig(output_plot, dpi=150)
    print(f"\nComparison plot saved successfully to: {os.path.abspath(output_plot)}")
    print("Backtesting workflow completed successfully!")

if __name__ == "__main__":
    main()
