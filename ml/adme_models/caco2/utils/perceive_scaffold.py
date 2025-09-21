import numpy as np
import seaborn as sns
from typing import Dict, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from utils import geometry, mol_utils

SUPER_FAMILIES: Dict = {
    "isoquinoline": {
        "smarts": "c1cc2c**cc2cn1",
        "transforms": [
            geometry.mirror(x=True),
            geometry.rotate(z=np.pi / 6),
        ],
    },
    "pyrrolopyrimidine": {
        "smarts": "c1cc2c*cnc2n1",
        "transforms": [geometry.mirror(x=True)],
    },
}

# NOTE: in python, dict are sorted by insertion order
# This is part of the spec of the language and guaranteed
CHEMICAL_SERIES: Dict[str, Dict] = {
    data["name"]: data
    for data in [
        {
            "SMARTS": "c1cc2c[#6,#7]cnc2n1",
            "name": "pyrrolopyrimidine",
            "super_family": "pyrrolopyrimidine",
            "subseries": {
                'pyrrolopyrimidine_5carboxamide': '[#7]-[#6](=O)-c1cnc2nc[#6,#7;a]c(-[#7])c12',
                'pyrrolopyrimidine_type2_a': '[#6]1-[#6]-[#6]-[#7](-[#6]C1)-c1[#6,#7;a]cnc2nccc12',
                'pyrrolopyrimidine_type2_b': '[#6]-1-[#6]-[#6]-[#7](-[#6]-1)-c1[#6,#7;a]cnc2nccc12',
                'pyrrolopyrimidine_type1': 'c1[#6,#7;a]cnc2nccc12',
                'pyrrolopyrimidine_other': 'c1nc2nc[#6,#7;a]cc2c1',
            },
        },
        {
            "SMARTS": "c1ccc2cc(ncc2c1)-c1nnc2[#6,#7;a][#6,#7;a][#6,#7;a][#6,#7;a]c12",
            "name": "isoquinolinyl-indazole",
            "super_family": "isoquinoline",
            "subseries": {
                '7-R-isoquinolinyl-indazole': 'c1cc2cnc(cc2c[#6,#7;a]1)-c1nnc2[#6,#7;a][#6,#7;a][#6,#7;a][#6,#7;a]c12',
                '1-R-isoquinolinyl-indazole': '[#6,#7;a]1cc2cnc(cc2cc1)-c1nnc2[#6,#7;a][#6,#7;a][#6,#7;a][#6,#7;a]c12',
            },
        },
        {
            "SMARTS": "c1ccc2cc(ncc2c1)-n1ncc2[#6,#7;a][#6,#7;a][#6,#7;a][#6,#7;a]c12",
            "name": "isoquinolinyl-reverse-indazole",
            "super_family": "isoquinoline",
            "subseries": {
                '7-R-isoquinolinyl-reverse-indazole': 'c1cc2cnc(cc2c[#6,#7;a]1)-n1ncc2[#6,#7;a][#6,#7;a][#6,#7;a][#6,#7;a]c12',
                '1-R-isoquinolinyl-reverse-indazole': '[#6,#7;a]1cc2cnc(cc2cc1)-n1ncc2[#6,#7;a][#6,#7;a][#6,#7;a][#6,#7;a]c12',
            },
        },
        {
            "SMARTS": "c1cc(nn1)-c1cc2c[#6,#7;a][#6,#7;a]cc2cn1",
            "name": "isoquinolinyl-pyrazole",
            "super_family": "isoquinoline",
            "subseries": {
                '1-R-Isoquinolinyl-4-oxo-pyrazole': '[#8]-c1cnnc1-c1cc2c[#6,#7;a]ccc2cn1',
                '1-R-Isoquinolinyl-4-amino-pyrazole': '[#7]-c1cnnc1-c1cc2c[#6,#7;a]ccc2cn1',
                '1-R-Isoquinolinyl-pyrazole': 'c1nnc(c1)-c1cc2c[#6,#7;a]ccc2cn1',
                '7-R-Isoquinolinyl-4-amino-pyrazole': '[#7]-c1cnnc1-c1cc2ccccc2cn1',
                '7-R-Isoquinolinyl-pyrazole': 'c1nnc(c1)-c1cc2ccccc2cn1',
            },
        },
        {"SMARTS": "c1ccc2ncncc2c1", "name": "quinazoline"},
        {"SMARTS": "c1cc2nc[#6,#7]cc2s1", "name": "thienopyrimidine"},
        {"SMARTS": "[#7]-c1nccc[#6,#7]1", "name": "2-aminopyrimidine"},
    ]
}

class Columns:
    """Namespace for static data"""

    # gloabl info
    structure = "SMILES"
    smiles = "SMILES"
    chemical_serie = 'Chemical series'
    chemical_subserie = "Chemical subseries"
    delivery_date = "delivery date"
    priority = 'project_management: Priority'

    # Experimental properties
    is_exp = "is_experimental"
    structure_exp = 'cddvault_mol (experimental)'
    charge_exp = "charge (experimental)"
    name_exp = 'Molecule Name (experimental)'
    synonym_exp = 'Synonyms'
    insilico_id_exp = "insilico ID (experimental)"

    # In-Silico properties
    is_insi = "is_insilico"
    structure_insi = 'cddvault_mol (in-silico)'
    charge_insi = 'charge (in-silico)'
    name_insi = 'Molecule Name (in-silico)'
    synonym_insi = 'Synonyms (in-silico)'
    aqemia_score = 'aqemia_score[protein=7l25]: aqemia_score (kcal/mol)'

    # Screenity properties
    structure_screenity = 'relaxation_mol'
    charge_screenity = 'charge (screenity)'
    net_molecular_charge = 'net_molecular_charge'

# Embed the superfamilies in 2D, with appropriate alignements
for name, data in SUPER_FAMILIES.items():
    mol = mol_utils.with_atomid(Chem.MolFromSmarts(data["smarts"]), offset=False)
    AllChem.Compute2DCoords(mol)
    for transform in data.get("transforms", []):
        mol = transform(mol)
    data["mol"] = mol

# Embed the chemical series SMARTS in 2D, aligned on the super_family if any
for data in CHEMICAL_SERIES.values():
    mol = Chem.MolFromSmarts(data["SMARTS"])
    template = SUPER_FAMILIES.get(data.get("super_family", ""), {}).get("mol")
    if template is not None:
        AllChem.GenerateDepictionMatching2DStructure(
            mol, template, allowRGroups=True, acceptFailure=False
        )
    else:
        AllChem.Compute2DCoords(mol)
    data["mol"] = mol

# Assign a color to each chemical serie
alpha = 0.3
palette = sns.color_palette("bright")
for i, data in enumerate(CHEMICAL_SERIES.values()):
    data["color"] = (*palette[i][:3], alpha)

# Compute subseries molecules object
for chemical_serie in CHEMICAL_SERIES.values():
    if "subseries" in chemical_serie:
        chemical_serie["subseries_mols"] = {
            name: Chem.MolFromSmarts(sma) for name, sma in chemical_serie["subseries"].items()
        }

def get_chemical_info(mol) -> Tuple[Optional[str], Optional[str]]:
    mol = mol_utils.as_mol(mol)
    for chemical_serie in CHEMICAL_SERIES.values():
        if mol.HasSubstructMatch(chemical_serie["mol"]):
            subseries = None
            if "subseries" in chemical_serie:
                for name, subserie_mol in chemical_serie["subseries_mols"].items():
                    if mol.HasSubstructMatch(subserie_mol):
                        subseries = name
                        break
            return chemical_serie["name"], subseries or chemical_serie["name"]
    return None, None