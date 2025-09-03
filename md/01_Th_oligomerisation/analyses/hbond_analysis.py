import scipy
import re

from contextlib import suppress

import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

from prolif.plotting.utils import separated_interaction_colors

import sklearn
from sklearn.mixture import GaussianMixture

import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis import dihedrals, distances, rms, align, contacts
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.analysis.data.filenames import Rama_ref

from waterdynamics import WaterOrientationalRelaxation as WOR
from waterdynamics import SurvivalProbability as SP

import prolif as plf

from rdkit import DataStructs

from analyses.utils.mda_utils import calculate_distances, add_data, _bit_to_color_value, _get_inv_color_mapper, _get_color_mapper

def run_hbond_analysis(in_silico_compound_ids, hbond_inputs, highlight_residues, simulation_path, overall_df, number_of_replicas, time_correction, simulation_length, residue_numbering_offset, d_a_cutoff, d_h_a_angle_cutoff, apolar_cutoff, ligand_torsions, use_reduced_traj):
    """Performs hbond and various structural analyses.

    Parameters
    ----------
    in_silico_compound_ids : list
        List of IAQ numbers to analyse.
    hbond_inputs : dict
        Details of which hbonds to track and how (H-A dist, COM, etc.)
    highlight_residues : dict
        Which residues to highlight in RMSF plot, some parameters (e.g. colour). Data is also used in other calculations.
    simulation_path : str
        Local store for simulations.
    overall_df : dict
        Overall dict to store all data.
    number_of_replicas : str
        Number of simulation replicas that can be analysed.
    time_correction : int
        Offset for plotting used when analysing a reduced trajectory. e.g. if you ran a 200 ns simulation resulting in 2000 frames but run analysis with a stride length of 200, this should be 10.
    simulation_length : int
        Length of each simulation (ns).
    residue_numbering_offset : int
        For plotting, as the residues have often been renumbered in the simulation relaive to the xtal structure.
    d_a_cutoff : float
        Distance cutoff for polar interactions (Å)
    d_h_a_angle_cutoff : int
        Angle cutoff for polar interactions (deg).
    apolar_cutoff : int
        Distance cutoff for apolar interactions (Å)
    use_reduced_traj : bool
        Bool to determine whether to run some calculatiosn on a reduced trajectory (defaults to True).

    Returns
    -------
    dict and mda Universe
        For use in plotting.
    """
    simulation_path = Path(simulation_path)
    for in_silico_compound_id in in_silico_compound_ids:
        # compound_path = simulation_path.joinpath(in_silico_compound_id) # TO CHANGE
        analysis_path = simulation_path.joinpath(f"analysis_{in_silico_compound_id}")
        # if not simulation_path.is_dir():
        #     print(f"{analysis_path} does not exist. Creating....")
        #     simulation_path.mkdir()
        for replica in range(number_of_replicas): #setup
            try:
                path_prmtop = simulation_path.joinpath(f"{in_silico_compound_id}_complex_structure_rep{replica}_system.prmtop")
                path_dcd = simulation_path.joinpath(f"{in_silico_compound_id}_complex_structure_rep{replica}_nopbc_traj.dcd") # use full trajectory    
                universe = mda.Universe(path_prmtop, path_dcd, format='DCD', topology_format='PRMTOP')
                universe_reference = mda.Universe(path_prmtop, path_dcd, format='DCD', topology_format='PRMTOP')
                
                if use_reduced_traj:
                    print("NOTE: using reduced trajectory for some analyses...")
                    stride_length = time_correction*10
                    system = universe.select_atoms('all')
                    system.write(f'{simulation_path}/{in_silico_compound_id}_rep{replica}_prot.dcd', frames=universe.trajectory[::stride_length])
                    path_dcd = simulation_path.joinpath(f"{in_silico_compound_id}_rep{replica}_prot.dcd") # use reduced trajectory    
                    universe = mda.Universe(path_prmtop, path_dcd, format='DCD', topology_format='PRMTOP')
                    universe_reference = mda.Universe(path_prmtop, path_dcd, format='DCD', topology_format='PRMTOP')
        
                ## RMSD        
                protein = universe.select_atoms('protein')
                ligand = universe.select_atoms("resname LIG")
                protein_c_alphas = universe.select_atoms('protein and name CA')
                resnames = protein.residues.resnames  # Array of residue names
                resids = protein.residues.resids      # Array of residue IDs
                num_frames = len(universe.trajectory)
                average = align.AverageStructure(universe, universe, select='protein and name CA',
                                        ref_frame=0).run()
                reference = average.results.universe
                aligner = align.AlignTraj(universe, reference,
                                select='protein and name CA',
                                in_memory=True).run()
                rmsd_analysis = rms.RMSD(universe, universe_reference, select="backbone", groupselections=["protein and not name H*", "resname LIG and not name H*"], ref_frame=0)
                rmsd_analysis.run()
                rmsd_df = pd.DataFrame(np.round(rmsd_analysis.results.rmsd[:, 2:], 2), columns = ['backbone', 'protein', 'ligand'])
                add_data(in_silico_compound_id, 'rmsd', rmsd_df, overall_df)
                # this is added to the numbering from VMD to have a correct number (like in the PDB and in papers). Please use the cell above to check that the residue_numbering_offset is correct
                ## RMSF
                # Perform RMSF analysis (per atom)
                rmsf_analysis = rms.RMSF(protein).run()
                rmsf_ca_analysis = rms.RMSF(protein_c_alphas).run()
                rmsf_ca_values = rmsf_ca_analysis.rmsf


                # Get the RMSF values for each atom
                rmsf_values = rmsf_analysis.rmsf

                # Initialize arrays for per-residue RMSF
                residue_rmsf = []

                # Calculate average RMSF per residue
                for residue in protein.residues:
                    # Get RMSF values for atoms in this residue using indices relative to the selection
                    atom_rmsf = rmsf_values[residue.atoms.indices - protein.atoms.indices[0]]
                    residue_rmsf.append(np.mean(atom_rmsf))  # Compute average RMSF for the residue
                add_data(in_silico_compound_id, 'rmsf', residue_rmsf, overall_df)
                add_data(in_silico_compound_id, 'rmsf_ca', rmsf_ca_values, overall_df)
                
                ## Hbond analysis

                protein_donors = "protein and (name N* or name O*) and bonded name H*"
                protein_acceptors = "protein and ((name N* and not bonded name H*) or name O*)"
                protein_hydrogens = "protein and name H* and bonded (name N* or name O*)"

                ligand_donors = "resname LIG and (name N* or name O*) and bonded name H*"
                ligand_acceptors = "resname LIG and ((name N* and not bonded name H*) or name O*)"
                ligand_hydrogens = "resname LIG and name H* and bonded (name N* or name O*)"
                hbond_analysis = HBA(
                    universe,
                    # donors_sel=f"({protein_donors}) or ({ligand_donors})",    # Both protein and ligand can be donors
                    hydrogens_sel=f"({protein_hydrogens}) or ({ligand_hydrogens})",    # Both protein and ligand can be donors
                    acceptors_sel=f"({protein_acceptors}) or ({ligand_acceptors})",  # Both protein and ligand can be acceptors
                    d_a_cutoff=d_a_cutoff, 
                    d_h_a_angle_cutoff=d_h_a_angle_cutoff,
                    between=['resname LIG', 'protein'],  # Only consider hydrogen bonds between protein and ligand
                    update_selections=False,
                )
                
                hbond_analysis.run()

                add_data(in_silico_compound_id, 'hbond', hbond_analysis, overall_df)

                hbonds = hbond_analysis.results.hbonds
                # Initialize a dictionary to store hydrogen bond counts
                hbond_dict = {}

                # Count occurrences of each hydrogen bond and map atom indices to residues
                for hbond in hbonds:
                    donor_atom = universe.atoms[int(hbond[1])]  # Donor atom
                    acceptor_atom = universe.atoms[int(hbond[3])]  # Acceptor atom
                    
                    # Handling donor
                    if donor_atom.resname == 'LIG':
                        donor_identifier = f"{donor_atom.resname}_{donor_atom.index}_{donor_atom.name}"  # Use atom index and atom name for LIG
                    else:
                        donor_identifier = f"{donor_atom.resname}_{donor_atom.resid + residue_numbering_offset}"  # Correct residue number

                    # Handling acceptor
                    if acceptor_atom.resname == 'LIG':
                        acceptor_identifier = f"{acceptor_atom.resname}_{acceptor_atom.index}_{acceptor_atom.name}"  # Use atom index and atom name for LIG
                    else:
                        acceptor_identifier = f"{acceptor_atom.resname}_{acceptor_atom.resid + residue_numbering_offset}"  # Correct residue number

                    key = (donor_identifier, acceptor_identifier)
                    if key in hbond_dict:
                        hbond_dict[key] += 1
                    else:
                        hbond_dict[key] = 1

                # Calculate occupancy for each hydrogen bond
                n_frames = len(universe.trajectory)  # Number of frames in the trajectory
                hbond_occupancy = {key: count / n_frames * 100 for key, count in hbond_dict.items()}

                # Convert the results into a DataFrame
                data = []
                for (donor_identifier, acceptor_identifier), occupancy in hbond_occupancy.items():
                    data.append({'Donor_Identifier': donor_identifier,
                                'Acceptor_Identifier': acceptor_identifier,
                                'Occupancy': occupancy})

                df = pd.DataFrame(data)
                # Sort the DataFrame by occupancy in descending order
                df_sorted = df.sort_values(by='Occupancy', ascending=True)

                # Create a new column for the donor-acceptor pairs
                df_sorted['Donor_Acceptor'] = df_sorted['Donor_Identifier'] + ' - ' + df_sorted['Acceptor_Identifier']
                add_data(in_silico_compound_id, 'hbond_pairs', df_sorted, overall_df)
                # Calculate distances with individual 'dist_to' options
                distance_dict_COM, distance_dict_closest, times = calculate_distances(universe, ligand, hbond_inputs, time_correction, residue_numbering_offset)
                add_data(in_silico_compound_id, 'distance_closest', distance_dict_closest, overall_df)
                
                ##### Commenting out hydrophobic analyses for now
                
                # hydrophobic_residues = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'PRO', 'TRP', 'CYS']
                # protein_hydrophobic_sidechains = universe.select_atoms(f"protein and (resname {' '.join(hydrophobic_residues)}) and not backbone")
                # ligand_hydrophobic = universe.select_atoms(f"resname LIG and name C*")  # Assuming ligand residue name is 'LIG'
                # apolar_cutoff=apolar_cutoff
                # hydrophobic_interactions = {}

                # for ts in universe.trajectory:
                #     frame_interactions = []
                    
                #     # Calculate COM for protein hydrophobic side chains (excluding backbone)
                #     for residue in protein_hydrophobic_sidechains.residues:
                #         # Compute COM for the side chain only (excluding backbone)
                #         side_chain_atoms = residue.atoms.select_atoms("not backbone")
                #         side_chain_com = side_chain_atoms.center_of_mass()
                        
                #         # Compare with each hydrophobic atom of the ligand
                #         for ligand_atom in ligand_hydrophobic:
                #             # Compute distance between the side chain COM and the ligand hydrophobic atom
                #             dist = np.linalg.norm(side_chain_com - ligand_atom.position)
                            
                #             # Check if the distance is within the cutoff
                #             if dist < apolar_cutoff:
                #                 interaction_details = {
                #                     "protein_resid": residue.resid,
                #                     "ligand_atom_index": ligand_atom.index,
                #                     "ligand_atom_name": ligand_atom.name,
                #                     "distance": dist
                #                 }
                #                 frame_interactions.append(interaction_details)
                    
                #     # Store interactions for this frame in the dictionary
                #     if frame_interactions:
                #         hydrophobic_interactions[ts.frame] = frame_interactions
                # # Dictionary to store the number of frames each residue has interactions (interaction occupancy)
                # interaction_occupancy = {}

                # # Loop through the hydrophobic_interactions dictionary to count unique residues per frame
                # for frame, interactions in hydrophobic_interactions.items():
                #     # Use a set to store unique residues for each frame (to avoid double counting)
                #     residues_in_frame = set()
                    
                #     for interaction in interactions:
                #         resid = interaction['protein_resid']
                #         shifted_resid = resid + residue_numbering_offset  # Shift the residue ID by +residue_numbering_offset
                #         residues_in_frame.add(shifted_resid)
                    
                #     # Update the interaction occupancy for each shifted residue
                #     for shifted_resid in residues_in_frame:
                #         if shifted_resid not in interaction_occupancy:
                #             interaction_occupancy[shifted_resid] = 0
                #         interaction_occupancy[shifted_resid] += 1

                # # Calculate occupancy percentage for each residue and store in a list for sorting
                # occupancy_list = []
                # for shifted_resid, count in interaction_occupancy.items():
                #     occupancy_percent = 100.0 * count / num_frames  # Normalize by the total number of frames
                #     occupancy_list.append((shifted_resid, occupancy_percent))

                # # Sort the list by occupancy percentage in descending order
                # sorted_occupancy_list = sorted(occupancy_list, key=lambda x: x[1], reverse=True)

                # resname_resid_list = []
                # occupancy_percent_list = []
                # for shifted_resid, occupancy_percent in sorted_occupancy_list:
                #     # Get the residue name for display (optional)
                #     ressel = universe.select_atoms(f"resid {shifted_resid - residue_numbering_offset}")  # Reverse the shift to get the correct resname
                #     resname = ressel.resnames[0] if len(ressel) > 0 else "UNK"  # Handle unknown residue cases
                #     resname_resid_list.append(f"{resname}{shifted_resid}")
                #     occupancy_percent_list.append(occupancy_percent)
                # df_hydrophobic = pd.DataFrame({'residue': resname_resid_list, 'occupancy': occupancy_percent_list})
                # add_data(in_silico_compound_id, 'hydrophobic', df_hydrophobic, overall_df)
                
                ### pocket analysis

                ### Ramachandran analysis
                
                ramachandran_phi_psi_dehidrals = dihedrals.Ramachandran(protein).run() # run Ramachandran analysis
                ramachandran_phi_psi_dehidrals_reshape = ramachandran_phi_psi_dehidrals.angles.reshape(
                    np.prod(ramachandran_phi_psi_dehidrals.angles.shape[:2]), 2)
                df_ramachandran = pd.DataFrame({'psi': ramachandran_phi_psi_dehidrals_reshape[:, 0], 'phi': ramachandran_phi_psi_dehidrals_reshape[:, 1]})
                add_data(in_silico_compound_id, 'ramachandran', df_ramachandran, overall_df)

                ### Q1/Q2 plots

                q1q2 = contacts.q1q2(universe, 'name CA', radius=8).run()

                q1q2_df = pd.DataFrame(q1q2.timeseries,
                        columns=['Frame',
                                    'Q1',
                                    'Q2'])

                q1q2_df = q1q2_df.melt(id_vars='Frame', value_vars=['Q1','Q2'])                    

                add_data(in_silico_compound_id, 'q1q2', q1q2_df, overall_df)

                ### Water analyses - water orientation restraint
                # Performed on entire trajectory

                path_dcd_full = simulation_path.joinpath(f"{in_silico_compound_id}_complex_structure_rep{replica}_nopbc_traj.dcd") # use full trajectory      
                universe_full = mda.Universe(path_prmtop, path_dcd_full, format='DCD', topology_format='PRMTOP')

                list_of_zones = [
                    f"byres resname HOH and sphzone 4.0 (protein and resid {highlight_residues['Cat lys'][0]})",
                    f"byres resname HOH and sphzone 4.0 (protein and resid {highlight_residues['C-loop start'][0]})",
                    f"byres resname HOH and sphzone 4.0 (protein and resid {highlight_residues['Pocket aC'][0]})",
                ]

                select_stab_water = f"byres resname HOH and sphzone 4.0 ((protein and resid {highlight_residues['Cat lys'][0]}) or (protein and resid {highlight_residues['C-loop start'][0]}))"

                WOR_analysis = WOR(universe_full, select_stab_water, 0, len(universe_full.trajectory), 20)
                WOR_analysis.run()

                wor_data = []

                time = 0 # TODO: replace counter 
                
                for WOR_OH, WOR_HH, WOR_dip in WOR_analysis.timeseries:
                    wor_data.append([time, WOR_OH, WOR_HH, WOR_dip])
                    time += 1

                wor_df = pd.DataFrame(wor_data, columns=['tau', 'WOR_OH', 'WOR_HH', 'WOR_dip'])
                # wor_df['time'] = wor_df['tau'] * time_correction
                wor_df = pd.melt(wor_df, id_vars='tau', value_vars=['WOR_OH', 'WOR_HH', 'WOR_dip'])
                
                add_data(in_silico_compound_id, 'wor', wor_df, overall_df)

                ### Water survival analysis (https://www.cell.com/biophysj/fulltext/S0006-3495(14)00601-8?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0006349514006018%3Fshowall%3Dtrue)

                sp_df = pd.DataFrame()

                sp_step = simulation_length/2 # setup
                    
                for zone in list_of_zones:
                    residue = re.search(r'\((.*?)\)', zone).group(1)
                    sp = SP(universe_full, zone, verbose=True)
                    sp.run(start=0, step=int(sp_step), intermittency=10) # Performed on every nth step, depending on how long the simulation is.
                    sp_df['tau'] = sp.tau_timeseries
                    sp_df[f'{residue}'] = sp.sp_timeseries
                
                # sp_df['time'] = sp_df['tau'] * time_correction
                sp_df = sp_df.melt(id_vars='tau', value_vars=[f"protein and resid {highlight_residues['Cat lys'][0]}", f"protein and resid {highlight_residues['C-loop start'][0]}", f"protein and resid {highlight_residues['Pocket aC'][0]}"])
                
                add_data(in_silico_compound_id, 'sr', sp_df, overall_df)
            
                ### Prolif contacts analysis

                protein_selection = universe.select_atoms("(protein or resname WAT) and byres around 20.0 group ligand", ligand=ligand)

                fp = plf.Fingerprint()
                fp.run(universe.trajectory, ligand, protein_selection)

                fp_df = fp.to_dataframe()

                add_data(in_silico_compound_id, 'fp_df', fp_df, overall_df)

                # Separate plot - Tanimoto similarity matrix

                # Tanimoto similarity matrix
                bitvectors = fp.to_bitvectors()
                similarity_matrix = []
                for bv in bitvectors:
                    similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(bv, bitvectors))
                frame_similarity_matrix = pd.DataFrame(similarity_matrix, index=fp_df.index, columns=fp_df.index)

                add_data(in_silico_compound_id, 'frame_similarity_matrix', frame_similarity_matrix, overall_df)
                                
                ### Loop distance matrices

                ploop_ca = universe.select_atoms(f"name CA and resid {highlight_residues['P-loop start'][0]}-{highlight_residues['P-loop end'][0]}")
                hinge_ca = universe.select_atoms(f"name CA and resid {highlight_residues['Hinge start'][0]}-{highlight_residues['Hinge end'][0]}")
                cloop_ca = universe.select_atoms(f"name CA and resid {highlight_residues['C-loop start'][0]}-{highlight_residues['C-loop end'][0]}")
                ac_ca = universe.select_atoms(f"name CA and resid {highlight_residues['aC start'][0]}-{highlight_residues['aC end'][0]}")

                n_ploop_ca = len(ploop_ca)
                n_hinge_ca = len(hinge_ca)
                n_cloop_ca = len(cloop_ca)
                n_ac_ca = len(ac_ca)

                print('P-loop has {} residues, hinge has {} residues and cloop has {} residues'.format(n_ploop_ca, n_hinge_ca, n_cloop_ca))

                dist_arr1 = distances.distance_array(ploop_ca.positions, # reference
                                        hinge_ca.positions, # configuration
                                        box=universe.dimensions)
                dist_arr2 = distances.distance_array(ploop_ca.positions, # reference
                                        cloop_ca.positions, # configuration
                                        box=universe.dimensions)
                dist_arr3 = distances.distance_array(ploop_ca.positions, # reference
                                        ac_ca.positions, # configuration
                                        box=universe.dimensions)                                       

                add_data(in_silico_compound_id, 'loop_dist_arr_ploop_hinge', [dist_arr1, ploop_ca, hinge_ca, n_ploop_ca, n_hinge_ca], overall_df)
                add_data(in_silico_compound_id, 'loop_dist_arr_ploop_cloop', [dist_arr2, ploop_ca, cloop_ca, n_ploop_ca, n_cloop_ca], overall_df)
                add_data(in_silico_compound_id, 'loop_dist_arr_ploop_ac_helix', [dist_arr3, ploop_ca, ac_ca, n_ploop_ca, n_ac_ca], overall_df)

                ### Torsion calculation - run on entire traj

                tor_atoms = ligand_torsions[in_silico_compound_id]

                dihedral_atoms = universe_full.select_atoms(f"resname LIG and name {tor_atoms[0]} {tor_atoms[1]} {tor_atoms[2]} {tor_atoms[3]}")
                dihedral = Dihedral([dihedral_atoms])

                dihedral.run()
                dihedral_angles = dihedral.angles
                dihedral_angles = [int(i) for i in dihedral_angles]
                
                # Convert to NumPy array and reshape for sklearn
                all_tors_np = np.array(dihedral_angles).reshape(-1, 1)
                
                add_data(in_silico_compound_id, 'torsions', [all_tors_np, tor_atoms], overall_df)    

            except: # needed in case there are simulations that didn't work so they will display as blank plots instead of crashing.
                continue

    return overall_df, protein

def plot_hbond_analysis(in_silico_compound_ids, hbond_inputs, highlight_residues, simulation_path, overall_df, number_of_replicas, time_correction, simulation_length, residue_numbering_offset, d_a_cutoff, d_h_a_angle_cutoff, apolar_cutoff, ligand_torsions, use_reduced_traj):
    """_summary_

    Parameters
    ----------
    in_silico_compound_ids : _type_
        _description_
    hbond_inputs : _type_
        _description_
    highlight_residues : _type_
        _description_
    simulation_path : _type_
        _description_
    overall_df : _type_
        _description_
    number_of_replicas : _type_
        _description_
    time_correction : _type_
        _description_
    simulation_length : _type_
        _description_
    residue_numbering_offset : _type_
        _description_
    d_a_cutoff : _type_
        _description_
    d_h_a_angle_cutoff : _type_
        _description_
    apolar_cutoff : _type_
        _description_
    use_reduced_traj : _type_
        _description_
    """
    overall_df, protein = run_hbond_analysis(
    in_silico_compound_ids=in_silico_compound_ids,
    highlight_residues=highlight_residues,
    hbond_inputs=hbond_inputs, 
    simulation_path=simulation_path, 
    overall_df=overall_df, 
    number_of_replicas=number_of_replicas, 
    time_correction=time_correction, 
    simulation_length=simulation_length,
    residue_numbering_offset=residue_numbering_offset,
    d_a_cutoff=d_a_cutoff, 
    d_h_a_angle_cutoff=d_h_a_angle_cutoff,
    apolar_cutoff=apolar_cutoff,
    ligand_torsions=ligand_torsions,
    use_reduced_traj=use_reduced_traj
)

    print("NOW PLOTTING THE DATA...")
    
    plot_step = 20 # arbitrary to set ticks on some plots TODO: make betterer
    plot_title_time = 'Time (ns)'

    for in_silico_compound_id in overall_df:
        simulation_path = Path(simulation_path)
        # compound_path = simulation_path.joinpath(in_silico_compound_id) # TO CHANGE
        # analysis_path = compound_path.joinpath(f"/analysis_{in_silico_compound_id}")
        fig1, axs1 = plt.subplots(3, 2, figsize=(20, 22))
        fig2, axs2 = plt.subplots(3, 2, figsize=(20, 22))
        fig3, axs3 = plt.subplots(3, 2, figsize=(20, 22))
        fig4, axs4 = plt.subplots(3, 2, figsize=(20, 22))
        fig5, axs5 = plt.subplots(3, 3, figsize=(20, 22))
        fig6, axs6 = plt.subplots(3, 2, figsize=(20, 22))
        fig7, axs7 = plt.subplots(3, 2, figsize=(20, 22))
        
        # Set titles at the top of each figure
        fig1.suptitle(f"1. RMSD / RMSF for {in_silico_compound_id}", y=1.01)
        fig2.suptitle(f"2. Polar h-bonds for {in_silico_compound_id}", y=1.01)
        fig3.suptitle(f"3. Distances and interaction matric for {in_silico_compound_id}", y=1.01)
        fig4.suptitle(f"4. Protein structural changes for {in_silico_compound_id}", y=1.01)
        fig5.suptitle(f"5. Loop distances for {in_silico_compound_id}", y=1.01)
        fig6.suptitle(f"6. Solvent orientational restraints and lifetimes for {in_silico_compound_id}", y=1.01)
        fig7.suptitle(f"7. Interaction fp similarity matrix and torsion frequency for {in_silico_compound_id}", y=1.01)

        for replica in range(number_of_replicas):
            # Extract data for current replica
            try:
                df_rmsd_replica = overall_df[in_silico_compound_id]['rmsd'][replica]
            except(IndexError):
                break
            rmsd_prot = df_rmsd_replica.protein.tolist()
            rmsd_lig = df_rmsd_replica.ligand.tolist()

            # RMSD Plot
            axs1[replica, 0].plot(rmsd_prot, label="Prot")
            axs1[replica, 0].plot(rmsd_lig, label="Lig")
            axs1[replica, 0].set_title("RMSD trace")
            axs1[replica, 0].set_ylim(-0.1, 6)
            axs1[replica, 0].set(xlabel=plot_title_time)
            axs1[replica, 0].set_ylabel("RMSD (Å)")
            axs1[replica, 0].grid(True)
            axs1[replica, 0].legend()
            axs1[replica, 0].set_xticks(np.arange(0, int(simulation_length/time_correction), step=1), labels=[str(int(i*time_correction+time_correction)) if i!=0 else 0 for i in range(int(simulation_length/time_correction))])
            if time_correction < 10:
                [l.set_visible(False) for (i,l) in enumerate(axs1[replica, 0].get_xticklabels()) if i % 5 != 0]

            # RMSF Plot
            residue_rmsf = overall_df[in_silico_compound_id]['rmsf'][replica]
            rmsf_prot_ca = overall_df[in_silico_compound_id]['rmsf_ca'][replica]
            residue_numbers = [res.resid + residue_numbering_offset for res in protein.residues]
            x_values = np.arange(len(rmsf_prot_ca)) + residue_numbering_offset + 2
            all_atoms_line, = axs1[replica, 1].plot(residue_numbers, residue_rmsf, label='All atoms')
            ca_line, = axs1[replica, 1].plot(x_values, rmsf_prot_ca, label='CA')
            
            # Highlight Residues
            legend_labels = []
            vertical_lines = []
            for label, (resid, color) in highlight_residues.items():
                index = residue_numbers.index(resid)
                rmsf_value = residue_rmsf[index]
                legend_labels.append(f'{label} ({rmsf_value:.2f})')
                vertical_lines.append(axs1[replica, 1].axvline(x=resid, color=color, linestyle='--'))

            axs1[replica, 1].set_xlabel('Residue Number')
            axs1[replica, 1].set_ylim(-0.1, 6)
            axs1[replica, 1].set_ylabel('RMSF (Å)')
            axs1[replica, 1].set_title('Per-Residue RMSF')
            axs1[replica, 1].grid(True)
            axs1[replica, 1].legend([all_atoms_line, ca_line] + vertical_lines, ['All atoms', 'CA'] + legend_labels, loc='upper right', fontsize=10)

            # Hydrogen Bond Count Plot
            hbond_analysis_replica = overall_df[in_silico_compound_id]['hbond'][replica]
            corrected_times = [t * time_correction for t in hbond_analysis_replica.times]
            axs2[replica, 0].plot(corrected_times, hbond_analysis_replica.count_by_time())
            axs2[replica, 0].set_title("Number of hydrogen bonds over time")
            axs2[replica, 0].set_xlabel(plot_title_time)
            axs2[replica, 0].set_ylabel(r"$N_{HB}$")

            # Donor-Acceptor Occupancy Plot
            hbond_pairs_replica = overall_df[in_silico_compound_id]['hbond_pairs'][replica]
            axs2[replica, 1].barh(hbond_pairs_replica['Donor_Acceptor'], hbond_pairs_replica['Occupancy'], color='skyblue')
            axs2[replica, 1].set_xlim(0, 100)
            axs2[replica, 1].set_ylabel('Donor-Acceptor Pairs')
            axs2[replica, 1].set_xlabel('Occupancy (%)')
            axs2[replica, 1].set_title('Occupancy for Donor-Acceptor Pairs')
            for index, value in enumerate(hbond_pairs_replica['Occupancy']):
                axs2[replica, 1].text(value + 1, index, f'{value:.1f}', va='center')

            # Ligand Distance Plot
            distance_dict_replica = overall_df[in_silico_compound_id]['distance_closest'][replica]
            for label, distances in distance_dict_replica.items():
                sns.regplot(x=corrected_times, y=distances, x_estimator=scipy.stats.sem(distances), order=3, scatter=False, ax=axs3[replica, 0], label=label)
            axs3[replica, 0].set_xlabel(plot_title_time)
            axs3[replica, 0].set_ylim(-0.1, 10)
            axs3[replica, 0].set_ylabel('Distance (Å)')
            axs3[replica, 0].set_title('Distance with Ligand (closest_atom)')
            axs3[replica, 0].grid(True)
            axs3[replica, 0].legend()

            #### Commenting out hydrophobic occupancy plot for now
            
            # # Hydrophobic Occupancy Plot
            # hydrophobic_replica = overall_df[in_silico_compound_id]['hydrophobic'][replica]
            # axs3[replica, 1].barh(hydrophobic_replica['residue'], hydrophobic_replica['occupancy'], color='skyblue')
            # axs3[replica, 1].set_xlabel('Occupancy (%)')
            # axs3[replica, 1].set_ylabel('Residue')
            # axs3[replica, 1].set_xlim(0, 100)
            # axs3[replica, 1].set_title('Hydrophobic Occupancy by Residue')
            # axs3[replica, 1].invert_yaxis()
            # for index, value in enumerate(hydrophobic_replica['occupancy']):
            #     axs3[replica, 1].text(value + 1, index, f'{value:.1f}', va='center')

            ### Interaction fingerprint
            
            fp_replica = overall_df[in_silico_compound_id]['fp_df'][replica]

            fp_replica_transposed = fp_replica.astype(np.uint8).T.apply(_bit_to_color_value, axis=1)

            n_frame_ticks = 10
            residues_tick_location = 'top'
            figsize = (8,10)
            dpi = 100

            COLORS: ClassVar[Dict[Optional[str], str]] = {
                None: "white",
                **separated_interaction_colors,
            }

            cmap = ListedColormap(list(COLORS.values()))

            subplots_kwargs = {} # placeholder if we want to add the ability to change plots later.
            subplots_kwargs.setdefault("figsize", figsize)
            subplots_kwargs.setdefault("dpi", dpi)

            tight_layout_kwargs = {} # placeholder if we want to add the ability to change plots later.
            tight_layout_kwargs.setdefault("pad", 1.2)

            im = axs3[replica, 1].imshow(
                fp_replica_transposed,
                aspect="auto",
                interpolation="none",
                cmap=cmap,
                vmin=0,
                vmax=max(_get_color_mapper().values()),
            )

            ## Frame ticks
            frames = fp_replica_transposed.columns
            max_ticks = len(frames) - 1
            
            for effective_n_ticks in (n_frame_ticks, n_frame_ticks - 1, n_frame_ticks + 1):
                samples, step = np.linspace(0, max_ticks, effective_n_ticks, retstep=True)
                if step.is_integer():
                    break
            else:
                samples = np.linspace(0, max_ticks, n_frame_ticks)
            indices = np.round(samples).astype(int)
            axs3[replica, 1].xaxis.set_ticks(indices, frames[indices])
            axs3[replica, 1].set_xlabel(plot_title_time)

            ## Residue ticks

            n_items = len(fp_replica_transposed.index)
            residues = fp_replica_transposed.index.get_level_values("protein")
            interactions = fp_replica_transposed.index.get_level_values("interaction")
            
            if residues_tick_location == "top":
                indices = [
                    i
                    for i in range(n_items)
                    if (i - 1 >= 0 and residues[i - 1] != residues[i]) or i == 0
                ]
            else:
                indices = [
                    i
                    for i in range(n_items)
                    if (i + 1 < n_items and residues[i + 1] != residues[i])
                    or i + 1 == n_items
                ]
            axs3[replica, 1].set_title('Barcode plot of interactions over time by type.')
            axs3[replica, 1].yaxis.set_ticks(indices, residues[indices])
            axs3[replica, 1].set_xticks(np.arange(0, simulation_length/time_correction, step=1), labels=[str(int(i*time_correction)) if i!=0 else 0 for i in range(int(simulation_length/time_correction))])
            [l.set_visible(False) for (i,l) in enumerate(axs3[replica, 1].get_xticklabels()) if i % 5 != 0]

            # Legend
            values: List[int] = np.unique(fp_replica_transposed.values).tolist()

            with suppress(ValueError):
                inv_color_mapper = _get_inv_color_mapper()
                # 0 not in values (e.g. plotting a single frame)
                values.pop(values.index(0))  # remove None color
            legend_colors = {
                inv_color_mapper[value]: im.cmap(value) for value in values
            }

            patches = [
                Patch(color=color, label=interaction)
                for interaction, color in legend_colors.items()
            ]

            axs3[replica, 1].legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
            
            #### Ramachandran plots
            ramachandran_replica = overall_df[in_silico_compound_id]['ramachandran'][replica]
            axs4[replica, 0].axis([-180, 180, -180, 180])
            axs4[replica, 0].axhline(0, color='k', lw=1)
            axs4[replica, 0].axvline(0, color='k', lw=1)
            axs4[replica, 0].set(xticks=range(-180, 181, 60), yticks=range(-180, 181, 60),
                xlabel=r"$\phi$", ylabel=r"$\psi$")
            degree_formatter = plt.matplotlib.ticker.StrMethodFormatter(
                r"{x:g}$\degree$")
            axs4[replica, 0].xaxis.set_major_formatter(degree_formatter)
            axs4[replica, 0].yaxis.set_major_formatter(degree_formatter)
            axs4[replica, 0].set_title('Ramachandran plot of backbone dihedral angles ($\psi$ / $\phi$)')
            axs4[replica, 0].scatter(ramachandran_replica.psi, ramachandran_replica.phi, s=20, c='black', alpha=0.5) # scatter of psi/phi angles
            X, Y = np.meshgrid(np.arange(-180, 180, 4),
                            np.arange(-180, 180, 4))
            levels = [1, 17, 15000]
            colors = ['#A1D4FF', '#35A1FF']
            axs4[replica, 0].contour(X, Y, np.load(Rama_ref), levels=levels, colors=colors) # contour plot of mda's Ramachandran reference data for comparison

            ### Native contacts analysis; Q1/Q2 plots - see https://www.pnas.org/doi/full/10.1073/pnas.1311599110
            
            q1q2_replica = overall_df[in_silico_compound_id]['q1q2'][replica]

            sns.lineplot(data = q1q2_replica,
                        x="Frame", y='value',hue="variable", ax=axs4[replica, 1]
                    )
            axs4[replica, 1].set_title('Native contacts analysis')
            axs4[replica, 1].set_ylabel('Fraction of native contacts (Q1/Q2)')
            axs4[replica, 1].set(xlabel=plot_title_time)
            axs4[replica, 1].set_xticks(np.arange(0, int(simulation_length/time_correction), step=1), labels=[str(int(i*time_correction+time_correction)) if i!=0 else 0 for i in range(int(simulation_length/time_correction))])
            if time_correction < 10:
                [l.set_visible(False) for (i,l) in enumerate(axs4[replica, 1].get_xticklabels()) if i % 5 != 0]
            handles, labels = axs4[replica, 1].get_legend_handles_labels()
            axs4[replica, 1].legend(handles=handles[0:], labels=labels[0:])
            
            ### loop distance heatmaps

            tick_interval=5
            
            loop_dist_arr_ploop_hinge_replica = overall_df[in_silico_compound_id]['loop_dist_arr_ploop_hinge'][replica]
            loop_dist_arr_ploop_cloop_replica = overall_df[in_silico_compound_id]['loop_dist_arr_ploop_cloop'][replica]
            loop_dist_arr_ploop_ac_helix_replica = overall_df[in_silico_compound_id]['loop_dist_arr_ploop_ac_helix'][replica]

            a=sns.heatmap(loop_dist_arr_ploop_hinge_replica[0], annot=False, fmt='.2f', square=True, ax=axs5[replica, 0], cmap='plasma', cbar_kws={'label': 'Distance (Angstrom)'})
            
            # p-loop - hinge distances
            # add residue ID labels to axes

            axs5[replica, 0].set_yticks(np.arange(loop_dist_arr_ploop_hinge_replica[3])[::tick_interval])
            axs5[replica, 0].set_xticks(np.arange(loop_dist_arr_ploop_hinge_replica[4])[::tick_interval])
            axs5[replica, 0].set_yticklabels(loop_dist_arr_ploop_hinge_replica[1].resids[::tick_interval])
            axs5[replica, 0].set_xticklabels(loop_dist_arr_ploop_hinge_replica[2].resids[::tick_interval])
            axs5[replica, 0].set_ylabel(f'{list(highlight_residues.keys())[0]} to {list(highlight_residues.keys())[1]}')
            axs5[replica, 0].set_xlabel(f'{list(highlight_residues.keys())[6]} to {list(highlight_residues.keys())[7]}')
            axs5[replica, 0].set_title('Distance between alpha-carbons')

            sns.heatmap(loop_dist_arr_ploop_cloop_replica[0], annot=False, fmt='.2f', square=False, ax=axs5[replica, 1], cmap='inferno', cbar_kws={'label': 'Distance (Angstrom)'})

            # p-loop - hinge distances
            # add residue ID labels to axes
            
            axs5[replica, 1].set_yticks(np.arange(loop_dist_arr_ploop_cloop_replica[3])[::tick_interval])
            axs5[replica, 1].set_xticks(np.arange(loop_dist_arr_ploop_cloop_replica[4])[::tick_interval])
            axs5[replica, 1].set_yticklabels(loop_dist_arr_ploop_cloop_replica[1].resids[::tick_interval])
            axs5[replica, 1].set_xticklabels(loop_dist_arr_ploop_cloop_replica[2].resids[::tick_interval])
            axs5[replica, 1].set_ylabel(f'{list(highlight_residues.keys())[0]} to {list(highlight_residues.keys())[1]}')
            axs5[replica, 1].set_xlabel(f'{list(highlight_residues.keys())[8]} to {list(highlight_residues.keys())[9]}')
            axs5[replica, 1].set_title('Distance between alpha-carbons')

            sns.heatmap(loop_dist_arr_ploop_ac_helix_replica[0], annot=False, fmt='.2f', square=True, ax=axs5[replica, 2], cmap='viridis', cbar_kws={'label': 'Distance (Angstrom)'})

            # p-loop - hinge distances
            # add residue ID labels to axes
            axs5[replica, 2].set_yticks(np.arange(loop_dist_arr_ploop_ac_helix_replica[3])[::tick_interval])
            axs5[replica, 2].set_xticks(np.arange(loop_dist_arr_ploop_ac_helix_replica[4])[::tick_interval])
            axs5[replica, 2].set_yticklabels(loop_dist_arr_ploop_ac_helix_replica[1].resids[::tick_interval])
            axs5[replica, 2].set_xticklabels(loop_dist_arr_ploop_ac_helix_replica[2].resids[::tick_interval])
            axs5[replica, 2].set_ylabel(f'{list(highlight_residues.keys())[0]} to {list(highlight_residues.keys())[1]}')
            axs5[replica, 2].set_xlabel(f'{list(highlight_residues.keys())[3]} to {list(highlight_residues.keys())[4]}')
            axs5[replica, 2].set_title('Distance between alpha-carbons')
        
            ### solvent WOR and lifetimes

            # wor
            
            wor_replica = overall_df[in_silico_compound_id]['wor'][replica]

            sns.lineplot(data = wor_replica,
                    x="tau", y='value',hue="variable", ax=axs6[replica, 0]
                )
            axs6[replica, 0].set(xlabel=plot_title_time)
            axs6[replica, 0].set(ylabel="P(wor)")
            axs6[replica, 0].set_title('Water orientational restraints (WOR).')
            handles, labels = axs6[replica, 0].get_legend_handles_labels()
            axs6[replica, 0].legend(handles=handles[0:], labels=labels[0:])
            axs6[replica, 0].set_xticks(np.arange(0, plot_step, step=1), labels=[str(int(i*(simulation_length/plot_step))) if i!=0 else 0 for i in range(int(plot_step))])
            [l.set_visible(False) for (i,l) in enumerate(axs6[replica, 0].get_xticklabels()) if i % 2 != 0]

            # Solvent lifetimes
            
            sr_replica = overall_df[in_silico_compound_id]['sr'][replica]
            
            sns.lineplot(data = sr_replica,
                x="tau", y='value',hue="variable", ax=axs6[replica, 1]
                )
            axs6[replica, 1].set(xlabel=plot_title_time)
            axs6[replica, 1].set(ylabel="P(SP)")
            axs6[replica, 1].set_title('Water survival probability by key residues.')
            handles, labels = axs6[replica, 1].get_legend_handles_labels()
            axs6[replica, 1].legend(handles=handles[0:], labels=labels[0:])
            axs6[replica, 1].set_xticks(np.arange(0, plot_step, step=1), labels=[str(int(i*(simulation_length/plot_step))) if i!=0 else 0 for i in range(int(plot_step))])
            [l.set_visible(False) for (i,l) in enumerate(axs6[replica, 1].get_xticklabels()) if i % 2 != 0]
        
            ### display heatmap of frame similarity matrices to compare binding mode changes.

            frame_similarity_matrix_replica = overall_df[in_silico_compound_id]['frame_similarity_matrix'][replica]

            sns.heatmap(
                frame_similarity_matrix_replica,
                ax=axs7[replica, 0],
                square=True,
                cmap='PRGn',
                vmin=0,
                vmax=1,
                center=0.5,
                xticklabels=5,
                yticklabels=5,
            )
            axs7[replica, 0].set_title('Tanimoto similarity matrix of simulation frames.')
            axs7[replica, 0].invert_yaxis()
            plt.yticks(rotation="horizontal")

            ### Torsion plots
            
            try:
                torsions_replica = overall_df[in_silico_compound_id]['torsions'][replica][0]
            
                tor_atoms = overall_df[in_silico_compound_id]['torsions'][replica][1]
                t1, t2, t3, t4 = tor_atoms[0], tor_atoms[1], tor_atoms[2], tor_atoms[3]
                
                # Fit a Gaussian Mixture Model with 2 components for plot
                gmm = GaussianMixture(n_components=2, random_state=0)
                gmm.fit(torsions_replica)
                means = gmm.means_.flatten()
                stds = np.sqrt(gmm.covariances_).flatten()
                weights = gmm.weights_.flatten()

                # Sort the components by mean for consistent labeling
                
                sorted_indices = np.argsort(means)
                means = means[sorted_indices]
                stds = stds[sorted_indices]
                weights = weights[sorted_indices]
                
                # Convert weights to percentages
                percentages = weights * 100

                # Plot the histogram
                
                g = sns.histplot(torsions_replica.flatten(), bins=36, binrange=(-180, 180),
                                                color='skyblue', alpha=0.7, edgecolor='black', label=in_silico_compound_id, ax=axs7[replica, 1])
                
                max_count = torsions_replica.flatten().max()

                # Overlay the means with vertical lines and annotate
                for idx, (mean, std, perc) in enumerate(zip(means, stds, percentages)):
                    color = 'red' if idx == 0 else 'green'
                    g.axvline(mean, color=color, linestyle='dashed', linewidth=2,
                                label=f'Distribution {idx+1} Mean')
                    # Position the text above max angle values and slightly above the histogram peak
                    y_position = max_count * 0.9 - idx * max_count * 0.1
                    g.text(mean + 5, y_position,
                        f'Mean: {mean:.2f}°\nStd: {std:.2f}°\n{perc:.2f}%',
                        color=color,
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.6))
                
                axs7[replica, 1].legend()
                axs7[replica, 1].set_ylabel('Frequency')
                axs7[replica, 1].set_xlabel('Torsion Angle (degrees)')
                axs7[replica, 1].set_title(f'Distribution of Torsion Angles ({t1}-{t2}-{t3}-{t4}) for {in_silico_compound_id}')
                axs7[replica, 1].grid(True, linestyle='--', alpha=0.5)
            except(KeyError):
                continue

        # plt.show()
        path_fig1 = simulation_path.joinpath(f"{in_silico_compound_id}_fig_1.png")
        fig1.tight_layout(rect=[0, 0.03, 1, 0.99])
        fig1.savefig(path_fig1)
        plt.close(fig1)
        path_fig2 = simulation_path.joinpath(f"{in_silico_compound_id}_fig_2.png")
        fig2.tight_layout(rect=[0, 0.03, 1, 0.99])
        fig2.savefig(path_fig2)
        plt.close(fig2)
        path_fig3 = simulation_path.joinpath(f"{in_silico_compound_id}_fig_3.png")
        fig3.tight_layout(rect=[0, 0.03, 1, 0.99])
        fig3.savefig(path_fig3)
        plt.close(fig3)
        path_fig4 = simulation_path.joinpath(f"{in_silico_compound_id}_fig_4.png")
        fig4.tight_layout(rect=[0, 0.03, 1, 0.99])
        fig4.savefig(path_fig4)
        plt.close(fig4)
        path_fig5 = simulation_path.joinpath(f"{in_silico_compound_id}_fig_5.png")
        fig5.tight_layout(rect=[0, 0.03, 1, 0.99])
        fig5.savefig(path_fig5)
        plt.close(fig5)
        path_fig6 = simulation_path.joinpath(f"{in_silico_compound_id}_fig_6.png")
        fig6.tight_layout(rect=[0, 0.03, 1, 0.99])
        fig6.savefig(path_fig6)
        plt.close(fig6)
        path_fig7 = simulation_path.joinpath(f"{in_silico_compound_id}_fig_7.png")
        fig7.tight_layout(rect=[0, 0.03, 1, 0.99])
        fig7.savefig(path_fig7)
        plt.close(fig7)