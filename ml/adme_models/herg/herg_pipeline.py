import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    matthews_corrcoef, f1_score, accuracy_score
)
import random
import torch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, NNConv
from torch_geometric.data import Data, Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rdkit import Chem
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# One-hot encoder with "unknown" category fallback

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]  # unknown
    return [int(x == s) for s in allowable_set]

def get_atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetSymbol(),
                                   ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H', 'B', 'Si', 'Se', 'As', 'Al', 'Zn', 'Cu', 'Ni', 'Fe', 'other'])  # 21
    features += one_hot_encoding(atom.GetHybridization(),
                                  [Chem.rdchem.HybridizationType.SP,
                                   Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3,
                                   Chem.rdchem.HybridizationType.SP3D,
                                   Chem.rdchem.HybridizationType.SP3D2,
                                   Chem.rdchem.HybridizationType.UNSPECIFIED])
    features += [atom.GetDegree()]  # Number of bonds
    features += [atom.GetFormalCharge()]
    features += [atom.GetNumRadicalElectrons()]
    features += [int(atom.GetIsAromatic())]
    return np.array(features, dtype=np.float32)

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    return np.array([
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ], dtype=np.float32)

def mol_to_graph(smiles: str, label: int = None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bond_feat, bond_feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data

# Use a GCN model, as it is 'good enough'.

class GCN(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=128):
        super().__init__()

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim * hidden_dim) 
        )

        self.conv1 = NNConv(input_dim, hidden_dim, self.edge_mlp, aggr='mean')

        self.edge_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_mlp2, aggr='mean')

        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(1)

# train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=42, stratify=[d.y.item() for d in data_list])

# train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_list, batch_size=32)

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch.y.float())
            total_loss += loss.item()
    return total_loss / len(loader)

def train_with_early_stopping(params, train_loader, val_loader):
    model = GCN(input_dim=params['input_dim'], edge_dim=params['edge_dim'], hidden_dim=params['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=params.get('pos_weight'))

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Load best model weights
    model.load_state_dict(torch.load('best_model.pth'))
    return model
