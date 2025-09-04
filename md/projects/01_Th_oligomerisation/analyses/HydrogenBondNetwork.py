import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as stats
import tqdm
from  MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from typing import List, Dict, Union, Any



class DirectHydrogenBonding:


    def __init__(self,
                 binding_site_resids: List[int],
                 ligand_resname: str,
                 start: int,
                 stop: int,
                 step: int,
                 u: str,
                 water_resname = 'HOH'):       
        

        self.binding_site_resids       = binding_site_resids
        self.ligand_resname            = ligand_resname
        self.u                         = u
        self.start,self.stop,self.step = start,stop,step 
        self.water_resname             = water_resname
        self._calculate_direct_hbonds()
        
    def _calculate_direct_hbonds(self):
        
  
        paths=[]

        binding_site_atoms = self.u.select_atoms('protein and resid '+str(' '.join(str(i) for i in self.binding_site_resids)))

        for ts in tqdm.tqdm(self.u.trajectory[self.start:self.stop:self.step]):

            hbonds_water = HBA(universe=self.u,
                              donors_sel='prop mass > 13 and resname '+str(self.ligand_resname)+' or  (prop mass > 13 and protein and  index '+str(' '.join(str(i) for i in binding_site_atoms.indices))+')',
                              hydrogens_sel='prop mass < 2 and resname '+str(self.ligand_resname)+' or  (prop mass < 2  and protein and  index '+str(' '.join(str(i) for i in binding_site_atoms.indices))+')',
                              acceptors_sel='prop mass > 13 and resname '+str(self.ligand_resname)+' or (prop mass > 13 and protein and  index '+str(' '.join(str(i) for i in binding_site_atoms.indices))+')',
                                between=['protein','resname '+str(self.ligand_resname)],
                              d_h_cutoff=1.2,
                              d_h_a_angle_cutoff=120,
                            d_a_cutoff=3.0
                            )
            
            hbonds_water.run(start=ts.frame,stop=ts.frame+1)

            unique_hbs_water=[list(i) for i in hbonds_water.hbonds[:,1:-2] if i[0]!=i[1]]
            
            G_water=nx.Graph()

            for edge in unique_hbs_water:

                if self.u.atoms.resnames[int(edge[0])]==self.water_resname:
                    donor=self.u.atoms.resnames[int(edge[0])]+str(self.u.atoms.resids[int(edge[0])])
                elif self.u.atoms.resnames[int(edge[0])]!=self.water_resname:
                    donor=self.u.atoms.resnames[int(edge[0])]+str(self.u.atoms.resids[int(edge[0])])+": "+str(self.u.atoms.names[int(edge[0])]) 
                if self.u.atoms.resnames[int(edge[2])]==self.water_resname:  
                    acceptor=self.u.atoms.resnames[int(edge[2])]+str(self.u.atoms.resids[int(edge[2])])
                elif self.u.atoms.resnames[int(edge[2])]!=self.water_resname:  
                    acceptor=self.u.atoms.resnames[int(edge[2])]+str(self.u.atoms.resids[int(edge[2])])+": "+str(self.u.atoms.names[int(edge[2])]) 
                G_water.add_edge(donor,acceptor)

            sources = [x for x in  G_water.nodes() if self.water_resname not in x and self.ligand_resname not in x]

            targets = [x for x in  G_water.nodes() if self.ligand_resname in x]
            
            for target in targets:

                if G_water.has_node(target):

                    for s in sources:

                        if nx.has_path(G_water, source=s, target=target)==True:

                            path = nx.all_shortest_paths(G_water, source=s, target=target)

                        # paths.append(list(path))

                        for p in path:

                            if len(p)==2:

                            ##check there's only one INX and one protein RES in the path
                            # if len({target}.intersection(set(p)))==1 and len(sources.intersection(set(p)))==1:

                                # print(p[0][0])
                                paths.append([list(p),ts.frame])
        
        self.paths = paths
        self.no_frames = len(self.u.trajectory[self.start:self.stop:self.step])
    # print(ts)
    def __call__(self):
        """
        
        """     
        



class WaterMediatedHydrogenBonding:


    def __init__(self,
                 binding_site_resids: List[int],
                 ligand_resname: str,
                 start: int,
                 stop: int,
                 step: int,
                 u: str,
                 water_resname = 'HOH'):       
        

        self.binding_site_resids       = binding_site_resids
        self.ligand_resname            = ligand_resname
        self.u                         = u
        self.start,self.stop,self.step = start,stop,step 
        self.water_resname             = water_resname
        self._calculate_wm_hbonds()
        
    def _calculate_wm_hbonds(self):
        
  
        paths=[]

        binding_site_atoms = self.u.select_atoms('protein and resid '+str(' '.join(str(i) for i in self.binding_site_resids)))

        for ts in tqdm.tqdm(self.u.trajectory[self.start:self.stop:self.step]):

            hbonds_water = HBA(universe=self.u,
                               donors_sel='prop mass > 13 and resname '+str(self.ligand_resname)+' or (prop mass > 13 and resname '+str(self.water_resname)+' and around 10 resname '+str(self.ligand_resname)+') or (prop mass > 13 and protein and  index '+str(' '.join(str(i) for i in binding_site_atoms.indices))+')',
                               hydrogens_sel='prop mass < 2 and resname '+str(self.ligand_resname)+' or (prop mass < 2 and resname '+str(self.water_resname)+' and around 10 resname '+str(self.ligand_resname)+') or (prop mass < 2  and protein and  index '+str(' '.join(str(i) for i in binding_site_atoms.indices))+')',
                               acceptors_sel='prop mass > 13 and resname '+str(self.ligand_resname)+' or (prop mass > 13 and resname '+str(self.water_resname)+' and around 10 resname '+str(self.ligand_resname)+') or (prop mass > 13 and protein and  index '+str(' '.join(str(i) for i in binding_site_atoms.indices))+')',
                               between=['protein or resname '+str(self.ligand_resname)+' ','resname '+str(self.water_resname)+''],
                               d_h_cutoff=1.2,
                               d_h_a_angle_cutoff=120,
                               d_a_cutoff=3.0
                            )
            hbonds_water.run(start=ts.frame,stop=ts.frame+1)


            unique_hbs_water=[list(i) for i in hbonds_water.hbonds[:,1:-2] if i[0]!=i[1]]
            G_water=nx.Graph()

            for edge in unique_hbs_water:

                if self.u.atoms.resnames[int(edge[0])]==self.water_resname:
                    donor=self.u.atoms.resnames[int(edge[0])]+str(self.u.atoms.resids[int(edge[0])])
                elif self.u.atoms.resnames[int(edge[0])]!=self.water_resname:
                    donor=self.u.atoms.resnames[int(edge[0])]+str(self.u.atoms.resids[int(edge[0])])+": "+str(self.u.atoms.names[int(edge[0])]) 
                if self.u.atoms.resnames[int(edge[2])]==self.water_resname:  
                    acceptor=self.u.atoms.resnames[int(edge[2])]+str(self.u.atoms.resids[int(edge[2])])
                elif self.u.atoms.resnames[int(edge[2])]!=self.water_resname:
                    acceptor=self.u.atoms.resnames[int(edge[2])]+str(self.u.atoms.resids[int(edge[2])])+": "+str(self.u.atoms.names[int(edge[2])]) 
                G_water.add_edge(donor,acceptor)

            sources = [x for x in  G_water.nodes() if str(self.ligand_resname) not in x and str(self.water_resname) not in x]

            targets = [x for x in  G_water.nodes() if str(self.ligand_resname) in x]
            
            
            for target in targets:

                if G_water.has_node(target):

                    for s in sources:

                        if nx.has_path(G_water, source=s, target=target)==True:

                            path = nx.all_shortest_paths(G_water, source=s, target=target)

                            # paths.append(list(path))

                            for p in path:

                                if len(p)==3:

                                ##check there's only one INX and one protein RES in the path
                                # if len({target}.intersection(set(p)))==1 and len(sources.intersection(set(p)))==1:

                                    # print(p[0][0])
                                    paths.append([list(p),ts.frame])

                                
    

        
        self.paths = paths
        self.no_frames = len(self.u.trajectory[self.start:self.stop:self.step])
    # print(ts)
    def __call__(self):
        """
        
        """     
        


