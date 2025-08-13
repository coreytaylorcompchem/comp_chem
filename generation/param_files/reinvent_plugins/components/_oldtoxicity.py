import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool

from rdkit import Chem

# from reinvent.models.molecule import Molecule
# from reinvent.reinvent_plugins.components import add_tag
from reinvent.scoring.score_components import BaseScoringComponent

# ----------------------------------------
# GNN Model Definition
# ----------------------------------------

component_type = "hergtoxicity"

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

# ----------------------------------------
# Feature Extractors
# ----------------------------------------

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

def get_atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H', 'other'])
    features += one_hot_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ])
    features += [atom.GetDegree()]
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

    if len(edge_index) == 0:
        return None  # skip molecules without bonds

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ----------------------------------------
# Scoring Component
# ----------------------------------------

# @add_tag("hergtoxicity")
class HERGToxicityComponent(BaseScoringComponent):
    def __init__(self, parameters: dict):
        from reinvent.scoring.score_components import BaseScoringComponent
        super().__init__(parameters)

        self.model_path = self.parameters.get("model_path", "/home/corey/Documents/comp_chem/generation/param_files/reinvent_plugins/scoring/herg_gnn/herg_gnn.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dummy values for input dimensions (from your code)
        self.input_dim = len(get_atom_features(Chem.MolFromSmiles("CC")[0]))
        self.edge_dim = len(get_bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))

        self.model = GCN(input_dim=self.input_dim, edge_dim=self.edge_dim)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def calculate_score(self, molecules: list[Molecule]) -> list[float]:
        if isinstance(molecules[0], str):
            smiles_list = molecules
        else:
            smiles_list = [mol.smiles for mol in molecules]
        graph_list = [mol_to_graph(smi) for smi in smiles_list]
        valid_graphs = [g for g in graph_list if g is not None]

        # Default to score = 0.0 for invalid graphs
        scores = [0.0 for _ in smiles_list]

        if not valid_graphs:
            return scores

        for i, graph in enumerate(graph_list):
            if graph is None:
                continue
            graph.batch = torch.tensor([0] * graph.x.size(0), dtype=torch.long)
            graph.to(self.device)
            with torch.no_grad():
                output = self.model(graph)
                prediction = torch.sigmoid(output).item()
                # Convert binary (0 = toxic, 1 = non-toxic)
                scores[i] = float(prediction < 0.5)

        return scores
