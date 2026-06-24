import pandas as pd
import numpy as np
import os

def generate_census(length, base_mean, daily_amp, weekly_amp, seed):
    """
    Generates realistic census data with daily and weekly patterns,
    autocorrelated noise (so census moves smoothly), and clips to [10, 20].
    """
    np.random.seed(seed)
    t = np.arange(length)
    
    # 1. Daily cycle (period is 4 steps/hour * 24 hours = 96 steps)
    # Peak in the late afternoon/evening, valley in the morning (discharges)
    daily = daily_amp * np.sin(2 * np.pi * t / 96 - np.pi / 3)
    
    # 2. Weekly cycle (period is 96 steps/day * 7 days = 672 steps)
    # Higher during weekdays, lower on weekends
    weekly = weekly_amp * np.sin(2 * np.pi * t / 672 - np.pi / 2)
    
    # 3. Autoregressive noise (AR-1) to simulate continuity
    noise = np.zeros(length)
    current_noise = 0
    for i in range(length):
        current_noise = 0.95 * current_noise + np.random.normal(0, 0.3)
        noise[i] = current_noise
        
    raw_census = base_mean + daily + weekly + noise
    
    # Round to integers (representing actual patients) and clip strictly to [10, 20]
    census = np.clip(np.round(raw_census), 10, 20).astype(int)
    return census

def main():
    # 30 days in June 2026
    start_time = "2026-06-01 00:00:00"
    end_time = "2026-06-30 23:45:00"
    
    # Frequency '15min' yields intervals of 15 minutes
    timestamps = pd.date_range(start=start_time, end=end_time, freq="15min")
    length = len(timestamps)
    
    # Generate data for Hospital 1 (3 Medsurg units)
    h1_medsurg_a = generate_census(length, base_mean=15.0, daily_amp=1.6, weekly_amp=1.2, seed=101)
    h1_medsurg_b = generate_census(length, base_mean=14.5, daily_amp=1.8, weekly_amp=1.0, seed=202)
    h1_medsurg_c = generate_census(length, base_mean=15.5, daily_amp=1.3, weekly_amp=1.5, seed=303)
    
    # Generate data for Hospital 2 (4 Medsurg units)
    h2_medsurg_a = generate_census(length, base_mean=16.0, daily_amp=1.4, weekly_amp=1.1, seed=404)
    h2_medsurg_b = generate_census(length, base_mean=14.0, daily_amp=1.7, weekly_amp=1.3, seed=505)
    h2_medsurg_c = generate_census(length, base_mean=15.0, daily_amp=1.5, weekly_amp=1.4, seed=606)
    h2_medsurg_d = generate_census(length, base_mean=14.8, daily_amp=1.2, weekly_amp=1.6, seed=707)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "H1_Medsurg_A": h1_medsurg_a,
        "H1_Medsurg_B": h1_medsurg_b,
        "H1_Medsurg_C": h1_medsurg_c,
        "H2_Medsurg_A": h2_medsurg_a,
        "H2_Medsurg_B": h2_medsurg_b,
        "H2_Medsurg_C": h2_medsurg_c,
        "H2_Medsurg_D": h2_medsurg_d,
    })
    
    csv_filename = "hospital_census.csv"
    df.to_csv(csv_filename, index=False)
    
    print("=== Hospital Census Generator ===")
    print(f"Time range: {start_time} to {end_time}")
    print(f"Total time steps: {length}")
    print(f"Data columns: {', '.join(df.columns)}")
    print(f"Successfully saved to: {os.path.abspath(csv_filename)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()
