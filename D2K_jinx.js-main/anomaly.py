from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler

# Constants
LATENT_DIM = 4
MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'vae_financial.pth')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

app = FastAPI()

class FinancialVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, latent_dim=LATENT_DIM):
        super(FinancialVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class FinancialDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def initialize_model_and_scaler(input_dim, data=None):
    """Initialize and save model and scaler if they don't exist"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Initialize or load model
    if not os.path.exists(MODEL_PATH):
        model = FinancialVAE(input_dim=input_dim)
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        model = FinancialVAE(input_dim=input_dim)
    
    # Initialize or load scaler
    if not os.path.exists(SCALER_PATH):
        scaler = StandardScaler()
        if data is not None:
            scaler.fit(data)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler

def detect_anomalies_in_data(financial_data, threshold_multiplier=1.0):
    """
    Detect anomalies in financial data using VAE
    """
    # Prepare data
    X = financial_data.select_dtypes(include=[np.number]).values
    
    # Initialize model and scaler
    model, scaler = initialize_model_and_scaler(input_dim=X.shape[1], data=X)
    
    # Transform data
    X_scaled = scaler.transform(X)
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    model.eval()
    
    # Create DataLoader
    test_dataset = FinancialDataset(X_scaled)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Detect anomalies
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            recon_batch, mu, log_var = model(batch)
            reconstruction_error = torch.mean((recon_batch - batch) ** 2, dim=1)
            reconstruction_errors.extend(reconstruction_error.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    threshold = np.mean(reconstruction_errors) + threshold_multiplier * np.std(reconstruction_errors)
    anomaly_indices = np.where(reconstruction_errors > threshold)[0]
    
    # Add results to dataframe
    financial_data['Reconstruction_Error'] = reconstruction_errors
    financial_data['Anomaly'] = 0
    financial_data.loc[anomaly_indices, 'Anomaly'] = 1
    
    # Return anomalies as JSON
    anomalies = financial_data[financial_data['Anomaly'] == 1].to_dict(orient='records')
    return {"num_anomalies": len(anomaly_indices), "anomalies": anomalies}

@app.post("/detect-anomalies/")
async def detect_anomalies_api(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    # Save and read CSV
    file_path = f"./uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    # Load CSV data
    financial_data = pd.read_csv(file_path)

    # Ensure required columns are present
    required_features = ['Income_Growth', 'Expenditure_Growth', 'PBT_Growth', 'Effective_Tax_Rate',
                          'EPS_Growth', 'FE_Earnings_Growth', 'FE_Outgo_Growth']

    if not all(feature in financial_data.columns for feature in required_features):
        raise HTTPException(status_code=400, detail=f"Missing required columns. Expected columns: {required_features}")

    # Detect anomalies
    result = detect_anomalies_in_data(financial_data[required_features])
    return result
