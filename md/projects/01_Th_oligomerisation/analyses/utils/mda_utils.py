import argparse
import os
import subprocess
import matplotlib as plt
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, ClassVar, Dict, List, Literal, Optional, Tuple

import MDAnalysis as mda
import svgutils.transform as sg
from IPython.display import SVG, display
from rdkit import Chem
from rdkit.Chem import rdDepictor
from matplotlib.colors import ListedColormap
from prolif.plotting.utils import separated_interaction_colors

def calculate_distances(universe, ligand, inputs, time_correction, fixed_number):
    """
    Calculate distances between the ligand and specified target atoms or COMs.
    
    Parameters:
    - universe: MDAnalysis Universe object
    - ligand: MDAnalysis AtomGroup object for the ligand
    - inputs: List of dictionaries specifying atom/residue targets and `dist_to` option
    
    Returns:
    - distance_dict_COM: Dictionary of distances for each label where dist_to='COM'
    - distance_dict_closest: Dictionary of distances for each label where dist_to='closest_atom'
    - times: Array of times corresponding to the distances
    """
    
    distance_dict_COM = {}
    distance_dict_closest = {}
    times = []

    # Loop over the trajectory and calculate the distance at each frame
    for ts in universe.trajectory:
        lig_com = ligand.center_of_mass()
        times.append(ts.time * time_correction)  # Record the time of the current frame

        # Loop through each input type
        for input_data in inputs:
            resid = input_data['resid']
            adjusted_resid = resid - fixed_number  # Adjust the residue ID to match the trajectory shift
            dist_to = input_data['dist_to']  # Get the dist_to method for this input
            use_hydrogen = input_data.get('use_hydrogen', False)  # Whether to use the bonded hydrogen
            dist = None
            resname = None
            label = ""

            try:
                # Retrieve residue name
                residue_selection = universe.select_atoms(f'resid {adjusted_resid}')
                if len(residue_selection) > 0:
                    resname = residue_selection.residues[0].resname

                # Handle different types of input
                if input_data['type'] == 'backbone_atom':
                    # Specific atom selection from backbone
                    atom_name = input_data['atom_name']
                    atom = universe.select_atoms(f'resid {adjusted_resid} and name {atom_name} and backbone')
                    
                    # Optionally use the bonded hydrogen atom
                    if use_hydrogen:
                        hydrogen = universe.select_atoms(f'resid {adjusted_resid} and name H and bonded name {atom_name}')
                        if len(hydrogen) > 0:
                            atom_position = hydrogen.positions[0]
                    else:
                        atom_position = atom.positions[0] if len(atom) > 0 else None
                    
                    if atom_position is not None:
                        dist = calculate_distance(ligand, atom_position, dist_to, lig_com)
                    label = f'{resname}{resid} {atom_name} (backbone {"H" if use_hydrogen else ""})'

                elif input_data['type'] == 'side_chain_atom':
                    # Specific atom selection from side chain
                    atom_name = input_data['atom_name']
                    atom = universe.select_atoms(f'resid {adjusted_resid} and name {atom_name} and not backbone')
                    if len(atom) > 0:
                        atom_position = atom.positions[0]
                        dist = calculate_distance(ligand, atom_position, dist_to, lig_com)
                    label = f'{resname}{resid} {atom_name} (side chain)'

                elif input_data['type'] == 'COM':
                    # Center of mass of the whole residue
                    residue = universe.select_atoms(f'resid {adjusted_resid}')
                    if len(residue) > 0:
                        res_com = residue.center_of_mass()
                        dist = calculate_distance(ligand, res_com, dist_to, lig_com)
                    label = f'{resname}{resid} COM'

                elif input_data['type'] == 'side_chain_COM':
                    # Center of mass of the side chain
                    residue = universe.select_atoms(f'resid {adjusted_resid} and not backbone')
                    if len(residue) > 0:
                        res_com = residue.center_of_mass()
                        dist = calculate_distance(ligand, res_com, dist_to, lig_com)
                    label = f'{resname}{resid} Side Chain COM'

                # Store the distances in a dictionary by the input label
                if dist is not None:
                    if dist_to == 'COM':
                        if label not in distance_dict_COM:
                            distance_dict_COM[label] = []
                        distance_dict_COM[label].append(dist)
                    elif dist_to == 'closest_atom':
                        if label not in distance_dict_closest:
                            distance_dict_closest[label] = []
                        distance_dict_closest[label].append(dist)

            except Exception as e:
                print(f"Error processing residue {resid} with input type {input_data['type']}: {str(e)}")

    # Convert times to a numpy array
    times = np.array(times)

    return distance_dict_COM, distance_dict_closest, times

def calculate_distance(ligand, target_position, dist_to, lig_com):
    """
    Calculate distance between the ligand and a target position (atom or COM).
    
    Parameters:
    - ligand: MDAnalysis AtomGroup object for the ligand
    - target_position: Coordinates of the target (atom or COM)
    - dist_to: 'COM' or 'closest_atom', indicating which distance to compute
    - lig_com: Center of mass of the ligand
    
    Returns:
    - Distance (float)
    """
    if dist_to == 'COM':
        # Distance to the ligand COM
        return np.linalg.norm(lig_com - target_position)
    elif dist_to == 'closest_atom':
        # Distance to the ligand's closest atom
        closest_atom_position = ligand.atoms.positions[np.argmin(np.linalg.norm(ligand.atoms.positions - target_position, axis=1))]
        return np.linalg.norm(closest_atom_position - target_position)

def plot_distances(times, distance_dict, dist_to_type):
    """
    Plot the distances over time.
    
    Parameters:
    - times: Array of times (ps)
    - distance_dict: Dictionary of distances with labels
    - dist_to_type: 'COM' or 'closest_atom', to be used in the plot title
    """
    # Plot distances for each input
    plt.figure(figsize=(10, 6))
    for label, distances in distance_dict.items():
        distances = np.array(distances)
        plt.plot(times, distances, label=label)

    # Customize the plot
    plt.xlabel('Time (ns)')
    plt.ylim(-0.1, 10)
    plt.ylabel('Distance (Å)')
    plt.title(f'Distance with Ligand ({dist_to_type})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

# Function to add a DataFrame for a specific sub-category and replica under a given pose_hash
def add_data(pose_hash, sub_category, data, overall_df):
    # Initialize the pose_hash key if it doesn't exist
    if pose_hash not in overall_df:
        overall_df[pose_hash] = {}
    
    # Initialize the sub-category key if it doesn't exist
    if sub_category not in overall_df[pose_hash]:
        overall_df[pose_hash][sub_category] = []
    
    # Append the replica DataFrame to the list for that sub-category
    overall_df[pose_hash][sub_category].append(data)

def _get_color_mapper():

    COLORS: ClassVar[Dict[Optional[str], str]] = {
            None: "white",
            **separated_interaction_colors,
    }

    color_mapper = {
        interaction: value for value, interaction in enumerate(COLORS)
    }
    return color_mapper

def _get_inv_color_mapper():
    
    color_mapper = _get_color_mapper()
    
    inv_color_mapper = {
                value: interaction for interaction, value in color_mapper.items()
    }

    return inv_color_mapper

def _bit_to_color_value(s: pd.Series) -> pd.Series:
    """Replaces a bit value with it's corresponding color value"""

    color_mapper = _get_color_mapper()
    
    interaction = s.name[-1]
    return s.apply(
        lambda v: (
            color_mapper[interaction] if v else color_mapper[None]
        ),
    )

def rdkit_labeled_with_mda_atom_nums(path, compound):
    u = mda.Universe(path + compound + "_complex_structure_rep0_system.prmtop", path + compound + "_complex_structure_rep0_traj.dcd")

    ligand_atoms = u.select_atoms("resname LIG and not element H")

    e = ligand_atoms.convert_to("RDKIT", NoImplicit=False)
    rdDepictor.Compute2DCoords(e)

    for i, j in zip(ligand_atoms, e.GetAtoms()):
        j.SetProp("atomNote", i.name)

    d = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(400, 400)  # or MolDraw2DSVG to get SVGs
    d.drawOptions().addAtomIndices = False
    d.drawOptions().baseFontSize = 1.0 # default is 0.6
    d.DrawMolecule(e)
    d.FinishDrawing()
    # svg to text
    s = d.GetDrawingText()
    s = s.replace('svg:','')
    # use svgutils to add text to svg
    fig = sg.fromstring(s)
    label = sg.TextElement(350,380, compound, size=14, 
                        font='sans-serif', anchor='middle', color='#119933')
    fig.append(label)
    svg = d.GetDrawingText()
    return SVG(fig.to_str())

def draw_mols_with_mda_atom_nums(path, compound_list):
    results = []
    for compound in compound_list:
        y = rdkit_labeled_with_mda_atom_nums(path, compound)
        results.append(y)
    for svg in results:
            display(svg)