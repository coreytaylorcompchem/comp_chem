import numpy as np

import rdkit.Chem as Chem

def extract_ligand_and_pocket_residues_within_distance(minimised_complex: str, distance_around_ligand = 6, protonated_acids = False):
    """Extract binding pocket and ligand within a specified distance around the ligand.

    Parameters
    ----------
    minimised_complex : str
        Complex stored locally.
    distance_around_ligand : int, optional
        Distance in Å from the ligand of residues you want to extract, by default 6
    protonated_acids : bool, optional
        Parameter to specify if you want acid groups protonated, by default False
    """    

    save_dir = minimised_complex.rpartition("/")[0]
    save_tag = minimised_complex.rpartition("/")[2].split('.')[0]
    
    pdb_complex_rdkit = Chem.MolFromPDBFile(minimised_complex, removeHs=False)

    # Extract ligand from PDB
    lig = list(filter(lambda frag: frag.GetNumAtoms()
        and frag.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName() == "LIG",
        Chem.GetMolFrags(pdb_complex_rdkit, asMols=True)))
    assert len(lig) == 1
    lig = lig.pop(0)

    # Get coord tuple arrays for the ligand and whole PDB complex
    residue_positions = lig.GetConformer().GetPositions()
    complex_positions = pdb_complex_rdkit.GetConformer().GetPositions()

    # Collect residue numbers within threshold
    residues_within_threshold = set()
    for i, complex_pos in enumerate(complex_positions):
        resnum = pdb_complex_rdkit.GetAtomWithIdx(i).GetPDBResidueInfo().GetResidueNumber()
        if resnum not in residues_within_threshold and closer_than(complex_pos, residue_positions, distance_around_ligand):
            residues_within_threshold.add(resnum)

    # Remove all atoms beyond threshold, making sure to flag acids as implicit
    with Chem.RWMol(pdb_complex_rdkit) as pdb_residues_within_threshold:
        for a in pdb_residues_within_threshold.GetAtoms():
            atom = a.GetPDBResidueInfo().GetName()
            if atom == ' OE2' or atom == ' OD2':
                if protonated_acids == False:
                    a.SetNoImplicit(True) # resets implicit hydrogen parameter for these atoms so we can add H's later
            else:
                a.SetNoImplicit(False)
            if a.GetPDBResidueInfo().GetResidueNumber() not in residues_within_threshold:
                pdb_residues_within_threshold.RemoveAtom(a.GetIdx())
                

    # Remove and add all implicit hydrogens, adding them back only to groups with unsatisfied valences, rather than adding to acids.
    # This is unless protonated_acids flag is set to True, then they will also be protonated.

    pdb_residues_within_threshold_with_h = Chem.RemoveHs(pdb_residues_within_threshold, implicitOnly=True)
    pdb_residues_within_threshold_with_h = Chem.AddHs(pdb_residues_within_threshold_with_h, addCoords=True)

    # Save residues within threshold
    Chem.MolToPDBFile(pdb_residues_within_threshold_with_h, f"{save_dir}/{save_tag}_bp.pdb")

def closer_than(atom_pos, residue_positions, threshold):
    """Return True if any coord tuple in residue_positions is closer than threshold to atom position."""
    return any(np.linalg.norm(atom_pos - pos) < threshold for pos in residue_positions)