# # feature_engineering.py

# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt

# def run_autoencoder(
#     file_path: str,
#     output_loss_plot: str = 'loss_curve.png',
#     output_xlsx: str = 'autoencoder_1_component_1_column_montecarlo_data.xlsx',
#     output_csv: str = 'encoded_1_component_montecarlo_data.csv',
#     epochs: int = 100,
#     batch_size: int = 64,
#     lr: float = 0.001
# ) -> pd.DataFrame:
#     """
#     Voert feature engineering uit via een eenvoudige Autoencoder op de 
#     (maandelijks geaggregeerde) NL-kolommen uit het datasetbestand.

#     Parameters
#     ----------
#     file_path : str
#         Pad naar de CSV met data (bijv. 'combined_data_wouter.csv').
#     output_loss_plot : str
#         Naam/path van het uit te schrijven verlies-plot (PNG).
#     output_xlsx : str
#         Naam/path van het uit te schrijven Excel-bestand (met eerste ge-encode kolom).
#     output_csv : str
#         Naam/path van het uit te schrijven CSV-bestand (met volledige encoding).
#     epochs : int
#         Aantal epochs voor training van de autoencoder.
#     batch_size : int
#         Batch-grootte voor de DataLoader.
#     lr : float
#         Learning rate voor de optimizer.
    
#     Returns
#     -------
#     encoded_df : pd.DataFrame
#         DataFrame met de ge-encode (gereduceerde) waarde(n) per maand (1 component).
#     """

#     # -----------------------
#     # 1. Data inladen
#     # -----------------------
#     data = pd.read_csv(file_path, sep=',')
#     # Zorg dat de 'time' kolom datetime is
#     data['time'] = pd.to_datetime(data['time'])

#     # Stel het aantal uren per (niet-schrikkel)jaar in
#     hours_per_year = 365 * 24
#     unique_patterns = len(data) // hours_per_year  # Aantal herhaalde jaren
    
#     # Voeg een 'unique_year' toe, startend vanaf 2019
#     data['unique_year'] = (data.index // hours_per_year) + 2019
#     data['time'] = data.apply(lambda row: row['time'].replace(year=row['unique_year']), axis=1)

#     # Extraheer year-month als period en groepeer
#     data['year_month'] = data['time'].dt.to_period('M')
#     nl_columns = [col for col in data.columns if col.startswith("NL")]
#     monthly_data = data.groupby('year_month')[nl_columns].mean()

#     # -----------------------
#     # 2. Schalen (StandardScaler)
#     # -----------------------
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(monthly_data)

#     # -----------------------
#     # 3. DataLoader opzetten
#     # -----------------------
#     scaled_data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
#     dataset = TensorDataset(scaled_data_tensor)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # -----------------------
#     # 4. Autoencoder-definitie
#     # -----------------------
#     class Autoencoder(nn.Module):
#         def __init__(self, input_dim, encoding_dim):
#             super(Autoencoder, self).__init__()
#             self.encoder = nn.Sequential(
#                 nn.Linear(input_dim, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, encoding_dim)
#             )
#             self.decoder = nn.Sequential(
#                 nn.Linear(encoding_dim, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, input_dim)
#             )

#         def forward(self, x):
#             encoded = self.encoder(x)
#             decoded = self.decoder(encoded)
#             return encoded, decoded

#     input_dim = scaled_data.shape[1]   # Aantal features (NL-kolommen)
#     encoding_dim = 1                   # Reduceren naar 1 dimensie

#     # -----------------------
#     # 5. Model, loss en optimizer
#     # -----------------------
#     model = Autoencoder(input_dim, encoding_dim)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # -----------------------
#     # 6. Trainen van de autoencoder
#     # -----------------------
#     losses = []
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         for batch in dataloader:
#             batch_data = batch[0]
#             optimizer.zero_grad()
#             encoded, decoded = model(batch_data)
#             loss = criterion(decoded, batch_data)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         avg_loss = epoch_loss / len(dataloader)
#         losses.append(avg_loss)
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

#     # -----------------------
#     # 7. Plot verlies-curve (optioneel)
#     # -----------------------
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(epochs), losses, label='Loss Curve', linewidth=2)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve Auto Encoding Wind Variable')
#     plt.legend()
#     plt.grid()
#     plt.savefig(output_loss_plot, format='png', dpi=300)
#     plt.show()

#     # -----------------------
#     # 8. Encodede data extraheren
#     # -----------------------
#     with torch.no_grad():
#         encoded_data, _ = model(scaled_data_tensor)

#     # In DataFrame gieten
#     encoded_df = pd.DataFrame(
#         encoded_data.numpy(),
#         columns=[f"Encoded_{i+1}" for i in range(encoding_dim)],
#         index=monthly_data.index.astype(str)
#     )

#     # Reset index voor bruikbaarheid
#     encoded_df = encoded_df.reset_index()
#     encoded_df.columns = ['year_month', 'encoded_1']

#     # Splits 'year_month' in jaar en maand
#     encoded_df[['year', 'month']] = encoded_df['year_month'].str.split('-', expand=True)
#     encoded_df['year'] = encoded_df['year'].astype(int)
#     encoded_df['month'] = encoded_df['month'].astype(int)
#     encoded_df.drop(columns=['year_month'], inplace=True)

#     # Voorbeeld: alleen de eerste encoded-kolom naar Excel wegschrijven
#     # (in dit geval is er maar 1 kolom).
#     montecarlo_df = encoded_df.iloc[:, 0]
#     montecarlo_df.to_excel(output_xlsx, index=False)

#     # Het volledige DataFrame met encodings naar CSV
#     encoded_df.to_csv(output_csv, index=False)

#     # Geef het resultaat terug indien je het verder wilt gebruiken
#     return encoded_df

# if __name__ == "__main__":
#     # Voorbeeld aanroep (pas paths aan naar jouw situatie)
#     df_encoded = run_autoencoder(
#         file_path="/kaggle/input/data-wouter-combined/combined_data_wouter.csv",
#         output_loss_plot="loss_curve.png",
#         output_xlsx="autoencoder_1_component_1_column_montecarlo_data.xlsx",
#         output_csv="encoded_1_component_montecarlo_data.csv",
#         epochs=100,
#         batch_size=64,
#         lr=0.001
#     )
#     print(df_encoded.head())

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def run_autoencoder(
    file_path: str,
    output_loss_plot: str = 'loss_curve.png',
    output_xlsx: str = 'autoencoder_1_component_1_column_montecarlo_data.xlsx',
    output_csv: str = 'encoded_1_component_montecarlo_data.csv',
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001
) -> pd.DataFrame:
    """
    Voert feature engineering uit via een eenvoudige Autoencoder op de 
    (maandelijks geaggregeerde) NL-kolommen uit het datasetbestand.

    Parameters
    ----------
    file_path : str
        Pad naar de CSV met data (bijv. 'combined_data_wouter.csv').
    output_loss_plot : str
        Naam/path van het uit te schrijven verlies-plot (PNG).
    output_xlsx : str
        Naam/path van het uit te schrijven Excel-bestand (met de ge-encode kolom).
    output_csv : str
        Naam/path van het uit te schrijven CSV-bestand (met de ge-encode kolom).
    epochs : int
        Aantal epochs voor training van de autoencoder.
    batch_size : int
        Batch-grootte voor de DataLoader.
    lr : float
        Learning rate voor de optimizer.
    
    Returns
    -------
    encoded_df : pd.DataFrame
        DataFrame met de ge-encode (gereduceerde) waarde(n) per maand (1 component).
    """

    # -----------------------
    # 1. Data inladen
    # -----------------------
    data = pd.read_csv(file_path, sep=',')
    data['time'] = pd.to_datetime(data['time'])

    # Stel het aantal uren per (niet-schrikkel)jaar in
    hours_per_year = 365 * 24
    unique_patterns = len(data) // hours_per_year  # Aantal herhaalde jaren
    
    # Voeg een 'unique_year' toe, startend vanaf 2019
    data['unique_year'] = (data.index // hours_per_year) + 2019
    data['time'] = data.apply(lambda row: row['time'].replace(year=row['unique_year']), axis=1)

    # Extraheer year-month als period en groepeer
    data['year_month'] = data['time'].dt.to_period('M')
    nl_columns = [col for col in data.columns if col.startswith("NL")]
    monthly_data = data.groupby('year_month')[nl_columns].mean()

    # -----------------------
    # 2. Schalen (StandardScaler)
    # -----------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(monthly_data)

    # -----------------------
    # 3. DataLoader opzetten
    # -----------------------
    scaled_data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    dataset = TensorDataset(scaled_data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -----------------------
    # 4. Autoencoder-definitie
    # -----------------------
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, encoding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    input_dim = scaled_data.shape[1]   # Aantal features (NL-kolommen)
    encoding_dim = 1                   # Reduceren naar 1 dimensie

    # -----------------------
    # 5. Model, loss en optimizer
    # -----------------------
    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -----------------------
    # 6. Trainen van de autoencoder
    # -----------------------
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch_data = batch[0]
            optimizer.zero_grad()
            encoded, decoded = model(batch_data)
            loss = criterion(decoded, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    # -----------------------
    # 7. Plot verlies-curve (optioneel)
    # -----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), losses, label='Loss Curve', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve Auto Encoding Wind Variable')
    plt.legend()
    plt.grid()
    plt.savefig(output_loss_plot, format='png', dpi=300)
    plt.show()

    # -----------------------
    # 8. Encodede data extraheren
    # -----------------------
    with torch.no_grad():
        encoded_data, _ = model(scaled_data_tensor)

    # In DataFrame gieten, met index = 'year_month' (als string)
    encoded_df = pd.DataFrame(
        encoded_data.numpy(),
        columns=[f"encoded_{i+1}" for i in range(encoding_dim)],
        index=monthly_data.index.astype(str)
    ).reset_index()

    # Hernoem de index-kolom
    encoded_df.rename(columns={'index': 'year_month'}, inplace=True)

    # (Optioneel: als je écht geen jaar/maand info meer wilt in de eindbestanden,
    # kan je deze regel overslaan en heb je alleen 'encoded_1'.)
    # Splits year-month als je het toch intern nodig hebt voor analyses:
    # encoded_df[['year', 'month']] = encoded_df['year_month'].str.split('-', expand=True)
    # encoded_df.drop(columns=['year_month'], inplace=True)

    # -----------------------
    # 9. Schrijf alleen de encoding weg naar Excel en CSV
    # -----------------------
    # Voor Excel: alleen de kolom 'encoded_1':
    encoded_df[['encoded_1']].to_excel(output_xlsx, index=False)
    
    # Voor CSV: idem alleen 'encoded_1':
    encoded_df[['encoded_1']].to_csv(output_csv, index=False)

    return encoded_df

if __name__ == "__main__":
    df_encoded = run_autoencoder(
        file_path="/kaggle/input/data-wouter-combined/combined_data_wouter.csv",
        output_loss_plot="loss_curve.png",
        output_xlsx="autoencoder_1_component_1_column_montecarlo_data.xlsx",
        output_csv="encoded_1_component_montecarlo_data.csv",
        epochs=100,
        batch_size=64,
        lr=0.001
    )
    print(df_encoded.head())

