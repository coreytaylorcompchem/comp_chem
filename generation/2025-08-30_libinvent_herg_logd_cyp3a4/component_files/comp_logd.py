from __future__ import annotations

import logging
import numpy as np
import torch

from rdkit import Chem, DataStructs
from torch_geometric.data import Batch, Data
from typing import List
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults, SmilesAssociatedComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles

logger = logging.getLogger("reinvent")

# --- MoleculeData class for correct batching ---
class MoleculeData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return super().__inc__(key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'global_features':
            return None
        return super().__cat_dim__(key, value)

# --- Descriptor + feature utils ---
from rdkit.Chem import Descriptors, Descriptors3D, rdPartialCharges, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np

descriptor_functions = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors,
    Descriptors.NumHAcceptors, Descriptors.TPSA, Descriptors.FractionCSP3,
    Descriptors.HeavyAtomCount, Descriptors.NumRotatableBonds, Descriptors.RingCount
]

electronegativity_dict = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19,
    16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
}

metals = {3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26,
          27, 28, 29, 30, 37, 38, 39, 40, 41, 42, 43, 44, 45,
          46, 47, 48, 49, 55, 56, 57, 72, 73, 74, 75, 76, 77,
          78, 79, 80, 81, 82, 83, 87, 88, 89, 104, 105, 106,
          107, 108, 109, 110, 111, 112, 113, 114, 115, 116}

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

def get_atom_features(atom, mol=None):
    pt = Chem.GetPeriodicTable()
    atomic_num = atom.GetAtomicNum()
    features = []

    features += one_hot_encoding(atom.GetSymbol(), [
        'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H', 'B',
        'Si', 'Se', 'As', 'Al', 'Zn', 'Cu', 'Ni', 'Fe', 'other'
    ])

    features += one_hot_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ])

    features += [
        atom.GetDegree(), atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(), int(atom.GetIsAromatic())
    ]

    features += [
        pt.GetAtomicWeight(atomic_num) / 200,
        pt.GetRvdw(atomic_num) / 2.5,
        electronegativity_dict.get(atomic_num, 0.0) / 4.0,
        pt.GetNOuterElecs(atomic_num) / 8.0,
        1.0 if atomic_num in metals else 0.0
    ]

    if mol:
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            charge = 0.0 if np.isnan(charge) or np.isinf(charge) else charge
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

def get_morgan_fingerprint(mol, radius=2, nBits=1024):
    generator = GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((nBits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def get_3d_descriptors(mol):
    mol = Chem.AddHs(mol)
    try:
        if Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG()) != 0:
            return np.zeros(6, dtype=np.float32)
        try:
            Chem.AllChem.UFFOptimizeMolecule(mol)
        except:
            return np.zeros(6, dtype=np.float32)
        descs = [
            Descriptors3D.Asphericity(mol),
            Descriptors3D.Eccentricity(mol),
            Descriptors3D.InertialShapeFactor(mol),
            Descriptors3D.SpherocityIndex(mol),
            rdMolDescriptors.CalcVolume(mol),
            rdMolDescriptors.CalcLabuteASA(mol)
        ]
        descs = [0.0 if np.isnan(d) or np.isinf(d) else d for d in descs]
        return np.array(descs, dtype=np.float32)
    except:
        return np.zeros(6, dtype=np.float32)

def mol_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except:
        pass

    x = torch.tensor([get_atom_features(atom, mol) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    global_feats = []
    for func in descriptor_functions:
        try:
            val = func(mol)
            val = 0.0 if np.isnan(val) or np.isinf(val) else val
        except:
            val = 0.0
        global_feats.append(val)

    fp = get_morgan_fingerprint(mol)
    desc3d = get_3d_descriptors(mol)

    global_features = torch.tensor(global_feats + fp.tolist() + desc3d.tolist(), dtype=torch.float32)

    return MoleculeData(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)

# --- GATv2 Model ---
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
import torch.nn.functional as F

class GATv2Regressor(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=128, heads=2, global_feat_dim=0):
        super().__init__()
        self.edge_encoder = torch.nn.Linear(edge_dim, hidden_dim)
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=0.1, edge_dim=hidden_dim)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=True, dropout=0.1, edge_dim=hidden_dim)

        # MLP to compress global features
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Final regressor head (combined GNN + global)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2), hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        device = next(self.parameters()).device

        x, edge_index, edge_attr, batch = x.to(device), edge_index.to(device), edge_attr.to(device), batch.to(device)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = global_mean_pool(x, batch)  # GNN output: [batch_size, hidden_dim]

        if hasattr(data, 'global_features'):
            global_features = data.global_features.to(x.device)  # [batch_size, global_feat_dim]
            g_feat = self.global_mlp(global_features)            # [batch_size, hidden_dim // 2]
            x = torch.cat([x, g_feat], dim=1)                    # Concatenate

        return self.lin(x).squeeze(1)


# --- Component class for REINVENT ---

@add_tag("__parameters")
@dataclass
class Parameters:
    model_file: List[str]


@add_tag("__component")
class LogD:
    def __init__(self, params: Parameters):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading LogD GATv2 model from {params.model_file[0]} on {device}")

        checkpoint = torch.load(params.model_file[0], map_location=device)

        self.model = GATv2Regressor(
            input_dim=checkpoint["input_dim"],
            edge_dim=checkpoint["edge_dim"],
            hidden_dim=checkpoint.get("hidden_dim", 128),
            heads=2,
            global_feat_dim=checkpoint.get("global_feat_dim", 9)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.smiles_type = "rdkit_smiles"

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> SmilesAssociatedComponentResults:
        scores = []

        for smi in smiles:
            data = mol_to_graph(smi)
            if data is None:
                logger.warning(f"Invalid SMILES: {smi}")
                scores.append(float('nan'))
                continue

            # data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            # data = data.to(self.device)

            # with torch.no_grad():
            #     score = self.model(data).item()
            
            data = Batch.from_data_list([data]).to(self.device)

            with torch.no_grad():
                score = self.model(data).item()

            logger.info(f"SMILES: {smi} | LogD Prediction: {score}")
            scores.append(float(score))

        return ComponentResults(scores=[np.array(scores)])

