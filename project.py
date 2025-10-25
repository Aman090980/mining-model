import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import streamlit as st


def load_csv(path):
   
    return pd.read_csv(path)


def merge_data(geo, chem, phys, sat, how='inner'):
    for df in [geo, chem, phys, sat]:
        if not {'lat', 'lon'}.issubset(df.columns):
            raise ValueError("Each dataset must contain 'lat' and 'lon' columns.")
    df = geo.merge(chem, on=['lat', 'lon'], how=how)
    df = df.merge(phys, on=['lat', 'lon'], how=how)
    df = df.merge(sat, on=['lat', 'lon'], how=how)
    return df


def preprocess(df):
  
    df = df.fillna(df.mean(numeric_only=True))
    scaler = StandardScaler()
    features = df.drop(columns=['lat', 'lon', 'label'], errors='ignore')
    df[features.columns] = scaler.fit_transform(features)
    return df


class MultiModalNet(nn.Module):
    def __init__(self, in_features, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze()


def train_model(model, X, y, epochs=20, lr=1e-3, device=None):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    X, y = X.to(device), y.to(device)

    criterion = nn.BCEWithLogitsLoss()
    y = torch.tensor(processed['label'].values, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    processed['label'] = processed['label'].map({
    'High': 1,
    'Medium': 0,
    'Low': 0
    }).astype(float)


    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    return model



def predict(model, X_new):
    model.eval()
    with torch.no_grad():
        preds = model(X_new)
    return preds.cpu().numpy()
def save_results(full_df, preds, output_path="prospectivity_results.csv"):
    results = full_df[['lat', 'lon', 'drilling_cost']].copy()
    results['score'] = preds
    results.to_csv(output_path, index=False)
    print(f"Results with drilling cost saved to {output_path}")
    return results

st.title(" Resource Exploration Prospectivity Dashboard")

uploaded = st.file_uploader("Upload Prospectivity Results CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader(" Top Ranked Prospects")
    st.dataframe(df.sort_values('score', ascending=False).head(10))

    st.subheader(" Exploration Map")
    st.map(df[['lat', 'lon']])
def feedback_loop(new_data, model):
    pass


def load_geological(path):
    return load_csv(path)

def load_geochemical(path):
    return load_csv(path)

def load_geophysical(path):
    return load_csv(path)

def load_satellite(path):
    return load_csv(path)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f" Using device: {device}")
df = load_csv("dummy_mining_data1.csv")
processed = preprocess(df)
print(f" Preprocessing complete. Final shape: {processed.shape}")

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
processed["label"] = encoder.fit_transform(processed["label"])


if 'label' not in processed.columns:
    raise KeyError(" 'label' column missing — cannot train without labels.")

X = torch.tensor(
    processed.drop(['lat', 'lon', 'label'], axis=1).values, dtype=torch.float32
).to(device)

y = torch.tensor(processed['label'].values, dtype=torch.float32).view(-1, 1).to(device)

print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

model = MultiModalNet(in_features=X.shape[1])
model = model.to(device)
print("Model initialized.")

train_model(model, X, y, epochs=10)
print("Training completed successfully.")

torch.save(model.state_dict(), "model.pth")
print("Model saved as 'model.pth'.")

preds = predict(model, X)

if torch.is_tensor(preds):
    preds = preds.detach().cpu().numpy().flatten()

latlon_df = df[['lat', 'lon','drilling_cost']].copy()
save_results(latlon_df, preds, output_path="prospectivity_results.csv")

print(" Predictions saved to 'prospectivity_results.csv'.")
df = pd.read_csv("prospectivity_results.csv")
print(df.head(20))
