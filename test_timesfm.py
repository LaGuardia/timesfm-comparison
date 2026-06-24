import os
import timesfm
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Starting TimesFM 2.5 test script...")
    
    # 1. Load the pretrained TimesFM 2.5 200M PyTorch model
    # Note: This will download the weights (~800MB) from Hugging Face if not already cached.
    print("Loading Google TimesFM 2.5 model weights from Hugging Face (google/timesfm-2.5-200m-pytorch)...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    
    # 2. Compile the model with forecast configuration
    print("Compiling model configuration...")
    context_len = 256
    horizon_len = 64
    
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=horizon_len,
            normalize_inputs=True,
        )
    )
    
    # 3. Create dummy time series data (sine wave + noise)
    print("Generating dummy time-series data...")
    t = np.linspace(0, 10, context_len)
    historical_data = np.sin(t) + np.random.normal(0, 0.1, context_len)
    
    # 4. Perform inference (Zero-shot forecasting)
    print(f"Running zero-shot forecasting for horizon of {horizon_len} steps...")
    # inputs is a list of time-series arrays
    forecast_results = model.forecast(
        horizon=horizon_len,
        inputs=[historical_data]
    )
    
    # forecast_results returns (point_forecast, experimental_quantile_forecast)
    # point_forecast has shape (number_of_series, horizon)
    point_forecast = forecast_results[0][0] 
    
    print("\n--- Forecast Completed ---")
    print(f"Input shape: {historical_data.shape}")
    print(f"Forecast shape: {point_forecast.shape}")
    print(f"Forecast (first 5 steps): {point_forecast[:5]}")
    
    # 5. Plot the result and save to disk
    plt.figure(figsize=(10, 5))
    
    # Plot historical data
    plt.plot(range(context_len), historical_data, label="Historical Data (Input)", color="blue")
    
    # Plot forecasted data
    forecast_range = range(context_len, context_len + horizon_len)
    plt.plot(forecast_range, point_forecast, label="Zero-Shot Forecast (TimesFM)", color="red", linestyle="--")
    
    plt.title("Google TimesFM 2.5 Zero-Shot Forecast Test")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    output_plot = "timesfm_test_plot.png"
    plt.savefig(output_plot)
    print(f"Plot saved successfully to: {os.path.abspath(output_plot)}")
    print("Script finished successfully!")

if __name__ == "__main__":
    main()
