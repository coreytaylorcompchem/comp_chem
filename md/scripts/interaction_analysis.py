from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
import numpy as np
import networkx as nx
from collections import defaultdict

class HydrogenBondCoordinationAnalyser:
    def __init__(self, universe, stride=1, hbond_cutoff=3.0, water_bridge_cutoff=4.0, first_sphere_cutoff=5.0):
        """
        Parameters:
            universe: MDAnalysis Universe object
            stride: trajectory frame stride for analysis
            hbond_cutoff: cutoff distance for coordination bonds (Å)
            water_bridge_cutoff: cutoff distance for water-mediated bridges (Å)
            first_sphere_cutoff: cutoff distance for water-water H-bonds near HETA (Å)
        """
        self.u = universe
        self.stride = stride
        self.hbond_cutoff = hbond_cutoff
        self.water_bridge_cutoff = water_bridge_cutoff
        self.first_sphere_cutoff = first_sphere_cutoff
        
        # Prepare atom selections
        self.heta_atoms = self.u.select_atoms("segid HETA")
        self.metal_sel = self.u.select_atoms("segid HETA")
        self.coord_atoms_hetb_sel = self.u.select_atoms("segid HETB and name O1 O2 O3 N1")
        self.coord_atoms_solv_sel = self.u.select_atoms("segid SOLV and name OH2")
        self.water_atoms_sel = self.u.select_atoms("segid SOLV and name OH2")

        # Initialize caches and containers
        self.coord_cache = {}
        self.graphs = {}
        self.water_water_near_heta = None
        
        # Placeholders for HBA object and hbonds data
        self.ghb = None
        self.frame_to_index = None

    def cache_coordinates(self):
        """Cache atomic coordinates for each frame according to the stride."""
        for ts in self.u.trajectory[::self.stride]:
            self.coord_cache[ts.frame] = {
                "metal": self.metal_sel.positions.copy(),
                "coord_hetb": self.coord_atoms_hetb_sel.positions.copy(),
                "coord_solv": self.coord_atoms_solv_sel.positions.copy(),
                "heta": self.heta_atoms.positions.copy(),
                "water": self.water_atoms_sel.positions.copy(),
                "time": ts.time,
            }

    def run_hbond_analysis(self):
        """Run hydrogen bond analysis on the trajectory."""
        self.ghb = HBA(
            universe=self.u,
            donors_sel="(segid HETA or segid HETB or segid SOLV) and name O1 O2 O3 OH2",
            hydrogens_sel="segid SOLV and name H1 H2",
            acceptors_sel="segid HETA or segid HETB or segid SOLV",
            d_a_cutoff=3.2,
            d_h_a_angle_cutoff=120.0,
        )
        self.ghb.run(start=0, stop=None, step=self.stride)

        # Map frames to indices for quick lookups
        unique_frames = np.unique(self.ghb.hbonds[:, 0].astype(int))
        self.frame_to_index = {frame: idx for idx, frame in enumerate(unique_frames)}

    def segment_pair_hbond_counts(self, segid1, segid2):
        """Compute hydrogen bond counts between two segments."""
        donor_indices = self.ghb.hbonds[:, 1].astype(int)
        acceptor_indices = self.ghb.hbonds[:, 3].astype(int)
        donor_segids = self.u.atoms[donor_indices].segids
        acceptor_segids = self.u.atoms[acceptor_indices].segids

        mask = ((donor_segids == segid1) & (acceptor_segids == segid2)) | \
               ((donor_segids == segid2) & (acceptor_segids == segid1))

        frame_indices = self.ghb.hbonds[mask, 0].astype(int)
        mapped_indices = np.array([self.frame_to_index[int(f)] for f in frame_indices]).astype(int)
        counts = np.bincount(mapped_indices, minlength=len(self.ghb.times))
        return counts

    def build_graphs(self):
        """Build networkx graphs per frame encoding H-bonds and coordination bonds."""
        metal_indices = self.metal_sel.indices
        hetb_indices = self.coord_atoms_hetb_sel.indices
        solv_indices = self.coord_atoms_solv_sel.indices
        heta_indices = self.heta_atoms.indices

        for frame in sorted(self.coord_cache):
            c = self.coord_cache[frame]
            G = nx.Graph()

            # Add nodes involved in hydrogen bonds at this frame
            hbonds_in_frame = self.ghb.hbonds[self.ghb.hbonds[:, 0] == frame]
            atom_indices = np.unique(np.concatenate([hbonds_in_frame[:, 1], hbonds_in_frame[:, 3]]).astype(int))
            for idx in atom_indices:
                atom = self.u.atoms[idx]
                G.add_node(idx, segid=atom.segid, name=atom.name)

            # Add hydrogen bond edges
            for hbond in hbonds_in_frame:
                donor_idx = int(hbond[1])
                acceptor_idx = int(hbond[3])
                G.add_edge(donor_idx, acceptor_idx, bond_type="H-bond", frame=frame, time=c["time"])

            # Add nodes for metals and coordination atoms if missing
            for idx in list(metal_indices) + list(hetb_indices) + list(solv_indices):
                if idx not in G:
                    atom = self.u.atoms[idx]
                    G.add_node(idx, segid=atom.segid, name=atom.name)

            # Add coordination bonds (metal - HETB)
            distances_hetb = np.linalg.norm(c["metal"][:, None, :] - c["coord_hetb"][None, :, :], axis=2)
            for i, metal_idx in enumerate(metal_indices):
                for j, hetb_idx in enumerate(hetb_indices):
                    if distances_hetb[i, j] < self.hbond_cutoff:
                        G.add_edge(metal_idx, hetb_idx, bond_type="Coordination", frame=frame, time=c["time"])

            # Add coordination bonds (metal - SOLV)
            distances_solv = np.linalg.norm(c["metal"][:, None, :] - c["coord_solv"][None, :, :], axis=2)
            for i, metal_idx in enumerate(metal_indices):
                for j, solv_idx in enumerate(solv_indices):
                    if distances_solv[i, j] < self.hbond_cutoff:
                        G.add_edge(metal_idx, solv_idx, bond_type="Coordination", frame=frame, time=c["time"])

            self.graphs[frame] = G

    def compute_water_water_near_heta(self):
        """Compute number of water-water hydrogen bonds near HETA per frame."""
        counts = np.zeros(len(self.ghb.times), dtype=int)
        frame_to_index = {int(round(t)): i for i, t in enumerate(self.ghb.times)}

        for hbond in self.ghb.hbonds:
            frame = int(hbond[0])
            donor_idx = int(hbond[1])
            acceptor_idx = int(hbond[3])
            donor = self.u.atoms[donor_idx]
            acceptor = self.u.atoms[acceptor_idx]

            if donor.segid == "SOLV" and acceptor.segid == "SOLV":
                pos_heta = self.coord_cache[frame]["heta"]
                d_donor = np.min(np.linalg.norm(pos_heta - donor.position, axis=1))
                d_acceptor = np.min(np.linalg.norm(pos_heta - acceptor.position, axis=1))
                if d_donor < self.first_sphere_cutoff or d_acceptor < self.first_sphere_cutoff:
                    frame_idx = frame_to_index.get(frame)
                    if frame_idx is not None:
                        counts[frame_idx] += 1
        self.water_water_near_heta = counts

    def get_solvation_switches(self, segid_central="HETA", segid_solvent="SOLV", bond_type="Coordination"):
        """
        Tracks the solvation shell of a central group over time and detects switches in solvent identity.
        
        Returns:
            shell_history: dict[frame] = set(atom indices of solvent in shell)
            switches: list of dicts with 'frame', 'joined', 'left' keys
        """
        shell_history = {}
        switches = []

        prev_shell = set()

        for frame in sorted(self.graphs.keys()):
            G = self.graphs[frame]
            current_shell = set()

            # Find solvent atoms bonded to the central group (e.g. HETA)
            for u_idx, v_idx, data in G.edges(data=True):
                if data.get("bond_type") != bond_type:
                    continue

                u_seg = G.nodes[u_idx]['segid']
                v_seg = G.nodes[v_idx]['segid']

                if (u_seg == segid_central and v_seg == segid_solvent):
                    current_shell.add(v_idx)
                elif (v_seg == segid_central and u_seg == segid_solvent):
                    current_shell.add(u_idx)

            shell_history[frame] = current_shell

            # Compare to previous shell
            if prev_shell:
                joined = current_shell - prev_shell
                left = prev_shell - current_shell

                switches.append({
                    "frame": frame,
                    "time": self.coord_cache[frame]["time"],
                    "joined": joined,
                    "left": left,
                    "net_change": len(joined) + len(left)
                })

            prev_shell = current_shell

        return shell_history, switches
    
    def get_persistent_coordinators(
        self,
        segid_central="HETA",
        segid_partner="SOLV",
        bond_type="Coordination",
        min_fraction=0.5,
        consecutive_only=False
    ):
        """
        Identify atoms (e.g., water) that are persistently coordinated to a central group.

        Args:
            segid_central: the central group (e.g., 'HETA')
            segid_partner: the coordinating atoms (e.g., 'SOLV')
            bond_type: edge type (e.g., 'Coordination')
            min_fraction: minimum fraction of frames to be considered persistent
            consecutive_only: if True, only consider uninterrupted sequences

        Returns:
            persistent_atoms: list of atom indices that meet criteria
            coordination_map: dict[atom_idx] = list of frames in which it's coordinated
        """

        def max_consecutive_streak(frames):
            """Return length of the longest consecutive sequence in a list of integers."""
            if not frames:
                return 0
            streak = max_streak = 1
            for i in range(1, len(frames)):
                if frames[i] == frames[i - 1] + 1:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 1
            return max_streak

        coordination_map = defaultdict(list)  # atom_idx -> list of frames
        frames = sorted(self.graphs.keys())

        for frame in frames:
            G = self.graphs[frame]
            for u_idx, v_idx, data in G.edges(data=True):
                if data.get("bond_type") != bond_type:
                    continue

                u_seg = G.nodes[u_idx]['segid']
                v_seg = G.nodes[v_idx]['segid']

                if (u_seg == segid_central and v_seg == segid_partner):
                    coordination_map[v_idx].append(frame)
                elif (v_seg == segid_central and u_seg == segid_partner):
                    coordination_map[u_idx].append(frame)

        persistent_atoms = []
        total_frames = len(frames)

        for atom_idx, frame_list in coordination_map.items():
            if consecutive_only:
                # Count longest consecutive streak
                streak = max_consecutive_streak(sorted(frame_list))
                if streak / total_frames >= min_fraction:
                    persistent_atoms.append(atom_idx)
            else:
                if len(frame_list) / total_frames >= min_fraction:
                    persistent_atoms.append(atom_idx)

        return persistent_atoms, coordination_map
    
    def get_residency_lifetimes(self, segid_central="HETA", segid_partner="SOLV", bond_type="Coordination"):
        """
        Calculate residency lifetimes of partner atoms coordinated to the central group.

        Returns:
            lifetimes_ps: list of residency times in picoseconds
            residency_map: dict[atom_idx] = list of (start_frame, end_frame, duration_ps)
        """

        coordination_map = defaultdict(list)  # atom_idx -> list of frames
        frames = sorted(self.graphs.keys())

        # Step 1: Build coordination map
        for frame in frames:
            G = self.graphs[frame]
            for u_idx, v_idx, data in G.edges(data=True):
                if data.get("bond_type") != bond_type:
                    continue

                u_seg = G.nodes[u_idx]['segid']
                v_seg = G.nodes[v_idx]['segid']

                if (u_seg == segid_central and v_seg == segid_partner):
                    coordination_map[v_idx].append(frame)
                elif (v_seg == segid_central and u_seg == segid_partner):
                    coordination_map[u_idx].append(frame)

        # Step 2: Convert frame lists into lifetimes
        residency_map = defaultdict(list)
        lifetimes_ps = []

        for atom_idx, coord_frames in coordination_map.items():
            if not coord_frames:
                continue
            coord_frames = sorted(coord_frames)

            # Segment into continuous blocks
            block_start = coord_frames[0]
            prev_frame = block_start

            for f in coord_frames[1:]:
                if f == prev_frame + self.stride:
                    prev_frame = f
                else:
                    # End current block
                    t0 = self.coord_cache[block_start]["time"]
                    t1 = self.coord_cache[prev_frame]["time"]
                    lifetime = t1 - t0
                    lifetimes_ps.append(lifetime)
                    residency_map[atom_idx].append((block_start, prev_frame, lifetime))
                    # Start new block
                    block_start = f
                    prev_frame = f

            # Handle last block
            t0 = self.coord_cache[block_start]["time"]
            t1 = self.coord_cache[prev_frame]["time"]
            lifetime = t1 - t0
            lifetimes_ps.append(lifetime)
            residency_map[atom_idx].append((block_start, prev_frame, lifetime))

        return lifetimes_ps, residency_map
    
    def get_edge_turnover(self, bond_type=None, normalize=False):
        """
        Compute edge turnover (formed + broken edges) per frame.
        
        Args:
            bond_type: str or None — restrict to e.g., 'H-bond' or 'Coordination'. If None, use all.
            normalize: bool — if True, return fraction of total edges at each frame.
            
        Returns:
            turnover_per_frame: list of (time_ps, turnover_count or turnover_fraction)
        """
        turnover_data = []
        frames = sorted(self.graphs.keys())
        
        prev_edges = set()

        for i, frame in enumerate(frames):
            G = self.graphs[frame]
            time = self.coord_cache[frame]["time"]
            
            # Filter edges by bond_type if needed
            edges = set()
            for u, v, data in G.edges(data=True):
                if bond_type is None or data.get("bond_type") == bond_type:
                    edge = tuple(sorted((u, v)))
                    edges.add(edge)

            if i == 0:
                turnover = 0  # No previous frame to compare
            else:
                added = edges - prev_edges
                removed = prev_edges - edges
                turnover = len(added) + len(removed)

            if normalize:
                denom = len(edges) if edges else 1  # Avoid division by zero
                turnover = turnover / denom

            turnover_data.append((time, turnover))
            prev_edges = edges

        return turnover_data
    
    def calculate_edge_addition_removal(self, bond_type="H-bond"):
        """
        Calculate number of edges added and removed between frames for a specific bond type.

        Returns:
            times (list): List of times corresponding to each frame (excluding first)
            added (list): Number of new edges at each frame
            removed (list): Number of edges that disappeared from previous frame
        """
        graphs = self.graphs  # Use the internal graphs
        sorted_frames = sorted(graphs.keys())
        prev_edges = set()
        
        times = []
        added = []
        removed = []

        for i, frame in enumerate(sorted_frames):
            G = graphs[frame]
            edges = {
                tuple(sorted((u, v)))
                for u, v, d in G.edges(data=True)
                if d.get("bond_type") == bond_type
            }

            if i == 0:
                prev_edges = edges
                continue

            # Calculate added and removed edges
            new_edges = edges - prev_edges
            gone_edges = prev_edges - edges

            added.append(len(new_edges))
            removed.append(len(gone_edges))
            times.append(self.coord_cache[frame]["time"])

            prev_edges = edges

        return times, added, removed
    
    def analyse_graph_topology(self, bond_type_filter=None):
        """
        Perform topological analysis on a dict of NetworkX graphs.

        Parameters:
            graphs (dict): frame -> nx.Graph
            bond_type_filter (str or None): if specified, only include edges of this bond type

        Returns:
            result (dict): {
                'frame': [...],
                'degree_centrality': [dict[node] = centrality],
                'clustering': [dict[node] = clustering coefficient],
                'n_components': [int],
                'component_sizes': [list of sizes]
            }
        """
        result = {
            "frame": [],
            "degree_centrality": [],
            "clustering": [],
            "n_components": [],
            "component_sizes": []
        }

        for frame in sorted(self.graphs):
            G = self.graphs[frame]

            if bond_type_filter:
                G_filtered = nx.Graph()
                for u, v, d in G.edges(data=True):
                    if d.get("bond_type") == bond_type_filter:
                        if u not in G_filtered:
                            G_filtered.add_node(u, **G.nodes[u])
                        if v not in G_filtered:
                            G_filtered.add_node(v, **G.nodes[v])
                        G_filtered.add_edge(u, v, **d)
            else:
                G_filtered = G.copy()

            deg_centrality = nx.degree_centrality(G_filtered)
            clustering = nx.clustering(G_filtered)
            components = list(nx.connected_components(G_filtered))
            n_components = len(components)
            component_sizes = [len(c) for c in components]

            result["frame"].append(frame)
            result["degree_centrality"].append(deg_centrality)
            result["clustering"].append(clustering)
            result["n_components"].append(n_components)
            result["component_sizes"].append(component_sizes)

        return result

    def run_all(self):
        """Convenience method to run the full analysis pipeline."""
        self.cache_coordinates()
        self.run_hbond_analysis()
        self.build_graphs()
        self.compute_water_water_near_heta()
        self.get_solvation_switches()
        self.get_persistent_coordinators()
        self.get_residency_lifetimes()
        self.get_edge_turnover()
        self.calculate_edge_addition_removal()
        # Save grap topology analysis results to self.
        self.topology_results_all = self.analyse_graph_topology()
        self.topology_results_hbond = self.analyse_graph_topology(bond_type_filter="H-bond")
        self.topology_results_coord = self.analyse_graph_topology(bond_type_filter="Coordination")

class InteractionAnalysis:
    def __init__(self, analyser):
        """
        analyser: instance of HydrogenBondCoordinationAnalyser
        """
        self.analyser = analyser
        self.frames = sorted(self.analyser.graphs.keys())
        self.times = [self.analyser.coord_cache[frame]["time"] for frame in self.frames]

        self.counts_hetb_solv = []
        self.coord_counts_hetb = []
        self.coord_counts_solv = []
        self.water_mediated_heta = []
        self.water_water_near_heta_graph = []

        # Precompute sets for quick lookup if needed (currently unused in this method)
        self.heta_indices_set = set(self.analyser.heta_atoms.indices)
        self.hetb_indices_set = set(self.analyser.coord_atoms_hetb_sel.indices)
        self.solv_indices_set = set(self.analyser.coord_atoms_solv_sel.indices)

    def run(self):
        for frame in self.frames:
            G = self.analyser.graphs[frame]
            
            hb_hetb_solv = 0
            coord_heta_hetb = 0
            coord_heta_solv = 0
            
            for u_idx, v_idx, data in G.edges(data=True):
                if data.get("frame") != frame:
                    continue
                
                bond_type = data.get("bond_type")
                u_segid = G.nodes[u_idx]['segid']
                v_segid = G.nodes[v_idx]['segid']

                if bond_type == "H-bond":
                    if (u_segid == "HETB" and v_segid == "SOLV") or (u_segid == "SOLV" and v_segid == "HETB"):
                        hb_hetb_solv += 1
                
                elif bond_type == "Coordination":
                    if (u_segid == "HETA" and v_segid == "HETB") or (u_segid == "HETB" and v_segid == "HETA"):
                        coord_heta_hetb += 1
                    elif (u_segid == "HETA" and v_segid == "SOLV") or (u_segid == "SOLV" and v_segid == "HETA"):
                        coord_heta_solv += 1
            
            # Water-mediated HETA↔HETB bridge calculation
            solv_nodes = [n for n, d in G.nodes(data=True) if d['segid'] == 'SOLV']
            heta_nodes = [n for n, d in G.nodes(data=True) if d['segid'] == 'HETA']
            hetb_nodes = [n for n, d in G.nodes(data=True) if d['segid'] == 'HETB']
            
            bridge_count = 0
            for solvn in solv_nodes:
                connected_to_heta = any(G.has_edge(solvn, hn) for hn in heta_nodes)
                connected_to_hetb = any(G.has_edge(solvn, hn) for hn in hetb_nodes)
                if connected_to_heta and connected_to_hetb:
                    bridge_count += 1
            
            self.water_mediated_heta.append(bridge_count)
            
            # Water-water H-bonds near HETA from graph edges and positions
            water_water_count = 0
            heta_positions = self.analyser.coord_cache[frame]['heta']
            
            for u_idx, v_idx, data in G.edges(data=True):
                if data.get("bond_type") == "H-bond":
                    u_seg = G.nodes[u_idx]['segid']
                    v_seg = G.nodes[v_idx]['segid']
                    if u_seg == 'SOLV' and v_seg == 'SOLV':
                        donor_pos = self.analyser.u.atoms[u_idx].position
                        acceptor_pos = self.analyser.u.atoms[v_idx].position
                        
                        d_donor = np.min(np.linalg.norm(heta_positions - donor_pos, axis=1))
                        d_acceptor = np.min(np.linalg.norm(heta_positions - acceptor_pos, axis=1))
                        
                        if d_donor < self.analyser.first_sphere_cutoff or d_acceptor < self.analyser.first_sphere_cutoff:
                            water_water_count += 1
                            
            self.water_water_near_heta_graph.append(water_water_count)
            
            self.coord_counts_hetb.append(coord_heta_hetb)
            self.coord_counts_solv.append(coord_heta_solv)
            self.counts_hetb_solv.append(hb_hetb_solv)
