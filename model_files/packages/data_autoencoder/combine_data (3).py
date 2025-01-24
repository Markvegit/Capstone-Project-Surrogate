# combine_data.py

import os
import pandas as pd
import zipfile

def extract_zip(zip_path, extract_to):
    """
    Extract a .zip-file to the specified directory.
    
    Parameters:
        zip_path (str): Pad naar het .zip-bestand.
        extract_to (str): Map waarin je de bestanden wilt uitpakken.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Bestanden uitgepakt naar: {extract_to}")

def combine_csv_files(directory, output_file):
    """
    Combine all CSV files in the specified directory into one CSV file.

    Parameters:
        directory (str): Pad naar de map met CSV-bestanden.
        output_file (str): Pad/naam van het CSV-bestand dat je wilt schrijven.
    """
    dataframes = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            print(f"Probeer in te lezen: {file_path}")
            try:
                df = pd.read_csv(file_path)
                dataframes.append(df)
            except Exception as e:
                print(f"Fout bij inlezen {file_path}: {e}")

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Alle CSV-bestanden succesvol samengevoegd in: {output_file}")
    else:
        print("Geen CSV-bestanden gevonden in de directory.")
