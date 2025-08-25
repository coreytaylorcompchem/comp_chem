import pandas as pd
from rdkit import Chem

def rdkit_descriptor(smiles, descriptor, docking_mol=None):
    if not pd.isnull(smiles):
        mol = Chem.MolFromSmiles(smiles)
    # elif not pd.isnull(docking_mol):
    #     mol = Chem.JSONToMols(docking_mol)[0]
    else:
        return None
    if descriptor == 'mw':
        rdkit_descriptor = Chem.Descriptors.MolWt(mol)
    if descriptor == 'tpsa':
        rdkit_descriptor = Chem.rdMolDescriptors.CalcTPSA(mol)
    if descriptor == 'logp':
        rdkit_descriptor = Chem.Descriptors.MolLogP(mol)
    if descriptor =='hba':
        rdkit_descriptor = Chem.Descriptors.NumHAcceptors(mol)
    if descriptor == 'hbd':
        rdkit_descriptor = Chem.Descriptors.NumHDonors(mol)
    if descriptor  == 'num_het_atoms':
        rdkit_descriptor = Chem.Descriptors.NumHeteroatoms(mol)
    if descriptor == 'num_rot_bonds':
        rdkit_descriptor = Chem.Descriptors.NumRotatableBonds(mol)
    if descriptor == 'num_heavy_atoms':
        rdkit_descriptor = Chem.Descriptors.HeavyAtomCount(mol)
    if descriptor == "num_aliphatic_carbocycles":
        rdkit_descriptor = Chem.Descriptors.NumAliphaticCarbocycles(mol)
    if descriptor == "num_aliphatic_carbocycles_mw":
        num = Chem.Descriptors.NumAliphaticCarbocycles(mol)
        mw = Chem.Descriptors.MolWt(mol)
        rdkit_descriptor = num / mw
    if descriptor == "num_aliphatic_heterocycles":
        rdkit_descriptor = Chem.Descriptors.NumAliphaticHeterocycles(mol)
    if descriptor == "num_aliphatic_heterocycles_mw":
        num = Chem.Descriptors.NumAliphaticHeterocycles(mol)
        mw = Chem.Descriptors.MolWt(mol)
        rdkit_descriptor = num / mw
    if descriptor == "num_aliphatic_rings":
        rdkit_descriptor = Chem.Descriptors.NumAliphaticRings(mol)
    if descriptor == "num_aromatic_carbocycles":
        rdkit_descriptor = Chem.Descriptors.NumAromaticCarbocycles(mol)
    if descriptor == "num_aromatic_heterocycles":
        rdkit_descriptor = Chem.Descriptors.NumAromaticHeterocycles(mol)
    if descriptor == "num_aromatic_rings":
        rdkit_descriptor = Chem.Descriptors.NumAromaticRings(mol)
    if descriptor == "num_aliphatic_rings_mw":
        num = Chem.Descriptors.NumAliphaticRings(mol)
        mw = Chem.Descriptors.MolWt(mol)
        rdkit_descriptor = num / mw
    if descriptor == "ring_count":
        rdkit_descriptor = Chem.Descriptors.RingCount(mol)
    if descriptor == "fraction_csp3":
        rdkit_descriptor = Chem.Descriptors.FractionCSP3(mol)
    if descriptor == 'npr1':
       rdkit_descriptor = Chem.rdMolDescriptors.CalcNPR1(mol)
    if descriptor == 'npr2':
       rdkit_descriptor = Chem.rdMolDescriptors.CalcNPR2(mol)
    if descriptor == 'inertial_shape':
        rdkit_descriptor = Chem.Descriptors3D.InertialShapeFactor(mol)
    if descriptor == "radius_gyration":
        rdkit_descriptor = Chem.Descriptors3D.RadiusOfGyration(mol)
    if descriptor == "charge":
        rdkit_descriptor = Chem.rdmolops.GetFormalCharge(mol)
    return rdkit_descriptor