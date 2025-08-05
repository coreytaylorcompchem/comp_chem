from typing import Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Lipinski import RotatableBondSmarts


def determine_contiguous_r_bond(mol):
    bond_groups = find_bond_groups(mol)
    largest_n_cont_rot_bonds = len(bond_groups[0]) if bond_groups else 0
    return largest_n_cont_rot_bonds


def add_descriptors(df_input: pd.DataFrame) -> pd.DataFrame:
    """Add necessary descriptors for SMILES filtering

    Args:
        df_input (pd.DataFrame): _description_

    Returns:
        pd.Dataframe: Df with additional columns (charge among them)
    """
    df = df_input.copy()
    df["mol_wt"] = df.apply(lambda x: MolWt(Chem.MolFromSmiles(x["SMILES"])), axis=1)
    df["highest_nbr_contiguous_rotatable_bonds"] = df.apply(
        lambda x: determine_contiguous_r_bond(Chem.MolFromSmiles(x["SMILES"])), axis=1
    )
    df["nbr_rotatable_bond"] = df.apply(
        lambda x: rdMolDescriptors.CalcNumRotatableBonds(
            Chem.MolFromSmiles(x["SMILES"]), strict=True
        ),
        axis=1,
    )
    df["nbr_hbond_donor"] = df.apply(
        lambda x: Lipinski.NumHDonors(Chem.MolFromSmiles(x["SMILES"])), axis=1
    )
    df["nbr_hbond_acceptor"] = df.apply(
        lambda x: Lipinski.NumHAcceptors(Chem.MolFromSmiles(x["SMILES"])), axis=1
    )
    df["largest_ring"] = df.apply(
        lambda x: find_largest_ring(Chem.MolFromSmiles(x["SMILES"])), axis=1
    )
    df["nbr_stereocenter"] = df.apply(
        lambda x: rdMolDescriptors.CalcNumAtomStereoCenters(Chem.MolFromSmiles(x["SMILES"])),
        axis=1,
    )
    df["is_zwitterion"] = df.apply(
        lambda x: is_zwitterion(Chem.MolFromSmiles(x["SMILES"])), axis=1
    )
    return df


def find_bond_groups(mol):
    """Find groups of contiguous rotatable bonds and return them sorted by decreasing size"""
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    rot_bond_set = set([mol.GetBondBetweenAtoms(*ap).GetIdx() for ap in rot_atom_pairs])
    rot_bond_groups = []
    while rot_bond_set:
        i = rot_bond_set.pop()
        connected_bond_set = set([i])
        stack = [i]
        while stack:
            i = stack.pop()
            b = mol.GetBondWithIdx(i)
            bonds = []
            for a in (b.GetBeginAtom(), b.GetEndAtom()):
                bonds.extend(
                    [
                        b.GetIdx()
                        for b in a.GetBonds()
                        if (
                            (b.GetIdx() in rot_bond_set)
                            and (not (b.GetIdx() in connected_bond_set))
                        )
                    ]
                )
            connected_bond_set.update(bonds)
            stack.extend(bonds)
        rot_bond_set.difference_update(connected_bond_set)
        rot_bond_groups.append(tuple(connected_bond_set))
    return tuple(sorted(rot_bond_groups, reverse=True, key=lambda x: len(x)))


def find_largest_ring(mol):
    try:
        largest_ring_size = max([len(x) for x in mol.GetRingInfo().AtomRings()])
        return largest_ring_size
    except Exception:
        return 0


def is_zwitterion(mol):
    chg = []
    for atom in mol.GetAtoms():
        chg.append(atom.GetFormalCharge())
    if sum(chg) == 0:
        diff_chg = set(chg)
        if len(diff_chg) != 1:
            return True
        else:
            return False
    else:
        return False


def update_names(mols_df):
    mols_df["idx"] = mols_df.groupby(["standard_inchikey"]).cumcount() + 1
    mols_df["idName"] = mols_df["idName"] + "_x" + mols_df["idx"].apply(lambda x: f"{x:05d}")
    return mols_df.drop(["idx"], axis=1)


def convert_str_to_float(mols_df):
    if "tautomer_distribution" in mols_df.columns:
        mols_df["tautomer_distribution"] = mols_df["tautomer_distribution"].astype(float)
    if "protonation_state_distribution" in mols_df.columns:
        mols_df["protonation_state_distribution"] = mols_df[
            "protonation_state_distribution"
        ].astype(float)
    return mols_df


def mol_to_json(mol: Union[Chem.Mol, str]):
    if mol and isinstance(mol, Chem.Mol):
        return Chem.MolToJSON(mol)
    if mol and isinstance(mol, str):
        return mol
    return None
