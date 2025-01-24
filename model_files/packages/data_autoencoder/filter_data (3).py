# filter_data.py

import pandas as pd

def filter_high_cost_data(input_file, output_file, std_factor=0.5, cost_col='Costs'):
    """
    Filter rows from an Excel file where the 'Costs' column is above
    (mean + std_factor * std_dev).
    Saves the filtered data to a new Excel file.

    Parameters:
        input_file (str): Path to the Excel file to read.
        output_file (str): Path to the Excel file to write the filtered data.
        std_factor (float): Factor of standard deviations above the mean used as the cutoff.
        cost_col (str): Column name for cost. Default='Costs'.
    """
    data = pd.read_excel(input_file)

    mean_costs = data[cost_col].mean()
    std_dev_costs = data[cost_col].std()
    threshold = mean_costs + std_factor * std_dev_costs

    filtered_data = data[data[cost_col] <= threshold]
    filtered_data.to_excel(output_file, index=False)
    print(f"Filtered data saved to '{output_file}'. "
          f"Original shape: {data.shape}, new shape: {filtered_data.shape}")
