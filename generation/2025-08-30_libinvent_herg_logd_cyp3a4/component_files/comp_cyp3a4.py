from __future__ import annotations

import logging
from typing import List
import numpy as np
import torch
from rdkit import Chem
from pydantic.dataclasses import dataclass
from torch_geometric.data import Data, Batch

from .component_results import ComponentResults, SmilesAssociatedComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles

logger = logging.getLogger("reinvent")

# --- MoleculeData for PyG ---
class MoleculeData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return super().__inc__(key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'global_features':
            return None
        return super().__cat_dim__(key, value)

# --- Feature engineering ---

descriptor_functions = [
    Chem.Descriptors.MolWt,
    Chem.Descriptors.MolLogP,
    Chem.Descriptors.NumHDonors,
    Chem.Descriptors.NumHAcceptors,
    Chem.Descriptors.TPSA,
    Chem.Descriptors.FractionCSP3,
    Chem.Descriptors.HeavyAtomCount,
    Chem.Descriptors.NumRotatableBonds,
    Chem.Descriptors.RingCount
]

electronegativity_dict = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19,
    16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
}

metals = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 55, 56, 57, 72,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    87, 88, 89, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116
}

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

def get_atom_features(atom, mol=None):
    pt = Chem.GetPeriodicTable()
    atomic_num = atom.GetAtomicNum()

    features = []
    features += one_hot_encoding(atom.GetSymbol(), [
        'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H',
        'B', 'Si', 'Se', 'As', 'Al', 'Zn', 'Cu', 'Ni', 'Fe', 'other'
    ])

    features += one_hot_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ])

    features += [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        int(atom.GetIsAromatic()),
        pt.GetAtomicWeight(atomic_num) / 200,
        pt.GetRvdw(atomic_num) / 2.5,
        electronegativity_dict.get(atomic_num, 0.0) / 4.0,
        pt.GetNOuterElecs(atomic_num) / 8.0,
        1.0 if atomic_num in metals else 0.0
    ]

    if mol:
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if np.isnan(charge) or np.isinf(charge):
                charge = 0.0
        except:
            charge = 0.0
        features.append(charge)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    return np.array([
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ], dtype=np.float32)

def mol_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    except:
        pass

    atom_features = [get_atom_features(atom, mol) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    global_features = []
    for func in descriptor_functions:
        try:
            val = func(mol)
            val = 0.0 if np.isnan(val) or np.isinf(val) else val
        except:
            val = 0.0
        global_features.append(val)
    global_features = torch.tensor(global_features, dtype=torch.float32)

    return MoleculeData(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)


# --- GIN Regressor Model ---
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn

class GINRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, global_feat_dim=9):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.lin = nn.Sequential(
            nn.Linear(hidden_dim + global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)

        if hasattr(data, 'global_features'):
            global_feats = data.global_features.to(x.device)
            x = torch.cat([x, global_feats], dim=1)

        return self.lin(x).squeeze(1)


# --- Component class for REINVENT ---

@add_tag("__parameters")
@dataclass
class Parameters:
    model_file: List[str]


@add_tag("__component")
class CYP3A4pIC50:
    def __init__(self, params: Parameters):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CYP3A4 GIN model from {params.model_file[0]} on {device}")

        checkpoint = torch.load(params.model_file[0], map_location=device)

        self.model = GINRegressor(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint.get("hidden_dim", 64),
            global_feat_dim=checkpoint.get("global_feat_dim", 9)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.smiles_type = "rdkit_smiles"

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> ComponentResults:
        scores = []

        for smi in smiles:
            data = mol_to_graph(smi)
            if data is None:
                logger.warning(f"Invalid SMILES: {smi}")
                scores.append(float('nan'))
                continue

            data = Batch.from_data_list([data]).to(self.device)

            with torch.no_grad():
                score = self.model(data).item()

            logger.info(f"SMILES: {smi} | CYP3A4 pIC50 Prediction: {score}")
            scores.append(float(score))

        return ComponentResults(scores=[np.array(scores)])
