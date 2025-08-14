# gcn_herg_component.py

import torch
import numpy as np
from rdkit import Chem

from reinvent_plugins.scoring.score_components.scoring_function_component import ScoringFunctionComponent
from reinvent_plugins.scoring.score_components.component_parameters import ComponentParameters

from .herg import GCN, mol_to_graph  # Replace with actual module


class GCNHergComponent(ScoringFunctionComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GCN(input_dim=21, edge_dim=6)
        self.model.load_state_dict(torch.load("/home/corey/Documents/comp_chem/ml/adme_models/herg/herg_gnn.pt/herg_gnn.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def score(self, smiles: list) -> np.ndarray:
        scores = []
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    scores.append(0.0)
                    continue

                data = mol_to_graph(smi, label=0)  # label is unused
                if data is None:
                    scores.append(0.0)
                    continue

                data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
                data = data.to(self.device)

                with torch.no_grad():
                    pred = self.model(data)
                    score = torch.sigmoid(pred).item()
                    scores.append(score)
            except Exception:
                scores.append(0.0)

        return np.array(scores)

