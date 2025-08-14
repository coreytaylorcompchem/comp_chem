from __future__ import annotations

import logging
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data

from rdkit import Chem

from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles

logger = logging.getLogger("reinvent")

# --- Utilities ---

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]  # unknown
    return [int(x == s) for s in allowable_set]

def get_atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetSymbol(),
                                ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H', 'B', 'Si', 'Se', 'As', 'Al', 'Zn', 'Cu', 'Ni', 'Fe', 'other'])
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

def mol_to_graph(smiles: str):
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

    # Dummy batch index for single graph
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


# --- GCN model ---

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


# --- REINVENT plugin component ---

@add_tag("__parameters")
@dataclass
class Parameters:
    model_file: List[str]  # Path to model checkpoint (only first is used here)


@add_tag("__component")
class Herg:
    def __init__(self, params: Parameters):
        # Load model checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading hERG GCN model from {params.model_file[0]} on {device}")
        checkpoint = torch.load(params.model_file[0], map_location=device)

        input_dim = checkpoint.get("input_dim", 30)  # fallback input size
        edge_dim = checkpoint.get("edge_dim", 6)    # fallback edge feature size

        self.model = GCN(input_dim=input_dim, edge_dim=edge_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.smiles_type = "rdkit_smiles"

    @normalize_smiles
    def __call__(self, smiles: List[str]):
        scores = []
        for smi in smiles:
            data = mol_to_graph(smi)
            if data is None:
                # Penalize invalid molecules
                scores.append(-1.0)
                continue

            data = data.to(self.device)
            with torch.no_grad():
                score = self.model(data).item()
            scores.append(score)

        return ComponentResults([(score,) for score in scores])

