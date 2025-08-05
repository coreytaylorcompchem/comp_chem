import base64
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.TemplateAlign import AlignMolToTemplate2D
from rdkit.Chem.Draw import rdMolDraw2D

def clear_sss_matches(mol_in):
    mol_in.__sssAtoms = []

class RGroupAligner:
    def __init__(self):
        self.ref_mol = Chem.MolFromSmarts("[#0]-[!#0]")
        Compute2DCoords(self.ref_mol)
        _ = self.ref_mol.GetConformer(0)
        
    def align(self,mol_in):
        Compute2DCoords(mol_in)
        _ = mol_in.GetConformer(0)
        AlignMolToTemplate2D(mol_in,self.ref_mol,clearConfs=True)

# Function to generate a molecule image as a base64 string
def generate_molecule_image(mol):
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)  # Image size (300x300)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    image_data = drawer.GetDrawingText()
    return base64.b64encode(image_data).decode("utf-8")

# Function to map a score to a heatmap color
def get_color(score, vmin=-1, vmax=1):
    cmap = plt.cm.coolwarm  # Use a heatmap color map
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # Normalize scores between 0 and 1
    rgba = cmap(norm(score))
    hex_color = mcolors.rgb2hex(rgba)  # Convert RGBA to hex color
    return hex_color