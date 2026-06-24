import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_file = "hospital_census.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run generate_hospital_data.py first.")
        return

    # Load data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Identify unit columns for each hospital
    h1_cols = [c for c in df.columns if c.startswith('H1_')]
    h2_cols = [c for c in df.columns if c.startswith('H2_')]
    
    # Calculate total census for each hospital
    df['H1_Total'] = df[h1_cols].sum(axis=1)
    df['H2_Total'] = df[h2_cols].sum(axis=1)
    
    # Set timestamp as index for plotting and rolling calculations
    df.set_index('timestamp', inplace=True)
    
    # Calculate 24-hour rolling averages (96 steps of 15 min = 24 hours)
    rolling_window = 96
    df_smoothed = df.rolling(window=rolling_window, center=True).mean()
    
    # Initialize the plot layout (3 rows, 1 column)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # --- Panel 1: Hospital Totals ---
    axes[0].plot(df.index, df['H1_Total'], label="Hospital 1 Total (Raw)", color='blue', alpha=0.15)
    axes[0].plot(df.index, df['H2_Total'], label="Hospital 2 Total (Raw)", color='green', alpha=0.15)
    axes[0].plot(df_smoothed.index, df_smoothed['H1_Total'], label="Hospital 1 Total (24h Smoothed)", color='blue', linewidth=2.5)
    axes[0].plot(df_smoothed.index, df_smoothed['H2_Total'], label="Hospital 2 Total (24h Smoothed)", color='green', linewidth=2.5)
    axes[0].set_title("Total Census per Hospital (June 2026)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Total Patients", fontsize=12)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # --- Panel 2: Hospital 1 Medsurg Units ---
    colors1 = ['#E74C3C', '#8E44AD', '#3498DB']
    for col, color in zip(h1_cols, colors1):
        axes[1].plot(df.index, df[col], label=f"{col} (Raw)", color=color, alpha=0.15)
        axes[1].plot(df_smoothed.index, df_smoothed[col], label=f"{col} (24h Smoothed)", color=color, linewidth=2.0)
    axes[1].set_title("Hospital 1 - Individual Medsurg Units", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Patients", fontsize=12)
    axes[1].legend(loc="upper left", ncol=3)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # --- Panel 3: Hospital 2 Medsurg Units ---
    colors2 = ['#2ECC71', '#F1C40F', '#1ABC9C', '#E67E22']
    for col, color in zip(h2_cols, colors2):
        axes[2].plot(df.index, df[col], label=f"{col} (Raw)", color=color, alpha=0.15)
        axes[2].plot(df_smoothed.index, df_smoothed[col], label=f"{col} (24h Smoothed)", color=color, linewidth=2.0)
    axes[2].set_title("Hospital 2 - Individual Medsurg Units", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("Patients", fontsize=12)
    axes[2].set_xlabel("Date", fontsize=12)
    axes[2].legend(loc="upper left", ncol=4)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_filename = "hospital_census_plot.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"Plot successfully saved to: {os.path.abspath(plot_filename)}")

if __name__ == "__main__":
    main()
