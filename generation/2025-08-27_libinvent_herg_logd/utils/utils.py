import base64
import pickle
import time
import os
from collections import defaultdict
from io import BytesIO
from typing import List, Tuple, Union
import logging

from tqdm import tqdm
from datetime import datetime
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from PIL import Image as pilImage
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem import (
    PandasTools,
    rdDepictor,
    rdFMCS,
    rdMolEnumerator,
    rdqueries,
    rdSubstructLibrary,
    rdTautomerQuery,
)
from rdkit.Chem.Draw import rdMolDraw2D

from utils import global_utils

logger = logging.getLogger(__name__)

def generate_search_library(df: pd.DataFrame, project_name: str):
    """Enumerates and generates a library of molecules to be queried with a substructure.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing RdKit Mol files
    project_name : str
        Name of project
    """

    suppl = df['rdkit_mols']
    RDLogger.DisableLog("rdApp.warning")
    t1 = time.time()
    data = []
    for i, mol in enumerate(suppl):
        if not ((i + 1) % 50000):
            print(f"Processed {i+1} molecules in {(time.time()-t1):.1f} seconds")
        if mol is None or mol.GetNumAtoms() > 50:
            continue
        fp = Chem.PatternFingerprint(mol, fpSize=1024, tautomerFingerprints=True)
        smi = Chem.MolToSmiles(mol)
        data.append((smi, fp))
    t2 = time.time()
    pickle.dump(data, open(f'{project_name}_ssdata.pkl', 'wb+'))
    t1 = time.time()
    mols = rdSubstructLibrary.CachedTrustedSmilesMolHolder()
    fps = rdSubstructLibrary.TautomerPatternHolder(1024)
    for smi, fp in data:
        mols.AddSmiles(smi)
        fps.AddFingerprint(fp)
    library = rdSubstructLibrary.SubstructLibrary(mols, fps)
    t2 = time.time()
    print(f"Generation took {t2-t1:.2f} seconds. The library has {len(library)} molecules.")
    pickle.dump(library, open(f'{project_name}_sslib.pkl', 'wb+'))


def get_aligned_queries(qry: Chem.Mol, tautomer_insensitive=True):
    """Enumerate molecules, convert each of the results into a TautomerQuery.

    Parameters
    ----------
    qry : Chem.Mol
        RDKit Mol
    tautomerInsensitive : bool, optional
        Tautomer insensitive search flag, by default False

    Returns
    -------
    List
        List containing molecules matching query substructure.
    """
    if not qry.GetNumConformers():
        rdDepictor.Compute2DCoords(qry)
    bndl = rdMolEnumerator.Enumerate(qry)

    # Find the MCS of the enumerated molecules:
    mcs = rdFMCS.FindMCS(bndl)
    qmcs = Chem.MolFromSmarts(mcs.smartsString)

    # Adjust query properties, generate coordinates, and create the TautomerQuery
    queries = []
    for query_db in bndl:
        query_db = Chem.AdjustQueryProperties(query_db)
        rdDepictor.GenerateDepictionMatching2DStructure(query_db, qry, refPatt=qmcs)
        if tautomer_insensitive:
            query_db = rdTautomerQuery.TautomerQuery(query_db)
        queries.append(query_db)
    return queries


def general_substructure_search(
    query: Union[str, Chem.Mol],
    sslib,
    tautomer_insensitive=True,  # Match different tautomers in the search library.
    alignResults=True,
    maxResults=1000,
) -> Union[str, Chem.Mol]:
    """Generate an RDKit SubstructLibrary, which allows more flexible searches across large
    databases (e.g. variable attachment points, tautomer insensitivity, link nodes, etc.),
    query the library and perform a substructure match.

    Parameters
    ----------
    query : Chem.Mol
        RDKit Mol
    sslib : Chem.rdSubstructLibrary.SubstructLibrary
        Substructure library object.
    tautomerInsensitive : bool, optional
        Tautomer insensitive search flag, by default False
    alignResults : bool, optional
        Align on coordinates (for display), by default True
    maxResults : int, optional
        Maximum number of results to retrieve from a search, by default 1000

    Returns
    -------
    List
        List containing molecules matching query substructures, index in database and matching atoms.
    """
    queries = get_aligned_queries(query, tautomer_insensitive=tautomer_insensitive)
    matches = []
    for q in queries:
        matches.extend(sslib.GetMatches(q, maxResults=maxResults))
    tmols = [(x, sslib.GetMol(x)) for x in matches]
    mols = []
    for idx, mol in sorted(tmols, key=lambda x: x[1].GetNumAtoms()):
        match = None
        if alignResults:
            for q in queries:
                if tautomer_insensitive:
                    match = q.GetSubstructMatch(mol)
                    if match:
                        rdDepictor.GenerateDepictionMatching2DStructure(
                            mol, q.GetTemplateMolecule()
                        )
                        break
                else:
                    match = mol.GetSubstructMatch(q)
                    if match:
                        rdDepictor.GenerateDepictionMatching2DStructure(mol, q)
                        break

        mols.append((idx, mol, match))
        if len(mols) >= maxResults:
            break
    return mols


def s3loadsdf(filepath: str, **kwargs) -> pd.DataFrame:
    """Load a mol saved as sdf and returns its associated dataframe.

    Parameters
    ----------
    filepath (str): s3 file path
        aws_profile (str): None if in AWS, on VM your profile name.

    Returns
    -------
    pd.DataFrame
        pd.DataFrame: Dataframe associated with query sdf.

    Raises
    ------
    AssertionError
        AssertionError: Wrong path on S3 error.
    """
    session = boto3.Session()
    s3 = session.client('s3')
    if not filepath.startswith("s3://") or not filepath.endswith('.sdf'):
        raise AssertionError('Not an sdf s3 path, please ensure your path begins with s3://.')
    filepath = filepath.lstrip('s3://')
    bucket = filepath.split('/')[0]
    key = "/".join(filepath.split('/')[1:])
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = PandasTools.LoadSDF(obj['Body'], smilesName='input_SMILES', **kwargs)
    return df


def highlight_rgroups(
    mol: Union[str, Chem.Mol],
    row: List[Chem.Mol],
    core: Union[str, Chem.Mol],
    width=350,
    height=200,
    fillRings=True,
    legend="",
    sourceIdxProperty="SourceAtomIdx",
    lbls=('R1'),
) -> bytes:
    """Generates aligned coordinates and creates images with highlighted R groups.

    Parameters
    ----------
    mol : Union[str, Chem.Mol]
        RDKit Mol
    row : List[Union[str, Chem.Mol]]
        List of all decomposed R-groups for a given molecule.
    core : Union[str, Chem.Mol]
        Core of molecule R-groups are attached to.
    width : int, optional
        Width of image for molecule plots, by default 350
    height : int, optional
        Height of image for molecule plots, by default 200
    fillRings : bool, optional
        Flag to determine whether highlights fill entire rings, by default True
    legend : str, optional
        Quantity to display as legend on molecule plots, by default ""
    sourceIdxProperty : str, optional
        Source atom property use for alignment, by default "SourceAtomIdx"
    lbls : tuple, optional
        R-group labels to display, by default ('R1')

    Returns
    -------
    bytes
        Byte array of converted png.
    """
    # copy the molecule and core
    mol = Chem.Mol(mol)
    core = Chem.Mol(core)

    # -------------------------------------------
    # include the atom map numbers in the substructure search in order to
    # try to ensure a good alignment of the molecule to symmetric cores
    for at in core.GetAtoms():
        if at.GetAtomMapNum():
            at.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(200 + at.GetAtomMapNum()))

    for lbl in row:
        if lbl == 'Core':
            continue
        rg = row[lbl]
        for at in rg.GetAtoms():
            if (
                not at.GetAtomicNum()
                and at.GetAtomMapNum()
                and at.HasProp('dummyLabel')
                and at.GetProp('dummyLabel') == lbl
            ):
                # attachment point. the atoms connected to this
                # should be from the molecule
                for nbr in at.GetNeighbors():
                    if nbr.HasProp(sourceIdxProperty):
                        mAt = mol.GetAtomWithIdx(nbr.GetIntProp(sourceIdxProperty))
                        if mAt.GetIsotope():
                            mAt.SetIntProp('_OrigIsotope', mAt.GetIsotope())
                        mAt.SetIsotope(200 + at.GetAtomMapNum())
    # remove unmapped hs so that they don't mess up the depiction
    rhps = Chem.RemoveHsParameters()
    rhps.removeMapped = False
    tmol = Chem.RemoveHs(mol, rhps)
    rdDepictor.GenerateDepictionMatching2DStructure(tmol, core)

    oldNewAtomMap = {}
    # reset the original isotope values and account for the fact that
    # removing the Hs changed atom indices
    for i, at in enumerate(tmol.GetAtoms()):
        if at.HasProp(sourceIdxProperty):
            oldNewAtomMap[at.GetIntProp(sourceIdxProperty)] = i
            if at.HasProp("_OrigIsotope"):
                at.SetIsotope(at.GetIntProp("_OrigIsotope"))
                at.ClearProp("_OrigIsotope")
            else:
                at.SetIsotope(0)

    # ------------------
    #  set up our colormap
    #   the three choices here are all "colorblind" colormaps

    # "Tol" colormap from https://davidmathlogic.com/colorblind
    colors = [
        (51, 34, 136),
        (17, 119, 51),
        (68, 170, 153),
        (136, 204, 238),
        (221, 204, 119),
        (204, 102, 119),
        (170, 68, 153),
        (136, 34, 85),
    ]
    # "IBM" colormap from https://davidmathlogic.com/colorblind
    colors = [(100, 143, 255), (120, 94, 240), (220, 38, 127), (254, 97, 0), (255, 176, 0)]
    # Okabe_Ito colormap from https://jfly.uni-koeln.de/color/
    colors = [
        (230, 159, 0),
        (86, 180, 233),
        (0, 158, 115),
        (240, 228, 66),
        (0, 114, 178),
        (213, 94, 0),
        (204, 121, 167),
    ]
    for i, x in enumerate(colors):
        # Ignore typing as, later, the modulus is calculated for colours to return ints.
        colors[i] = tuple(y / 255 for y in x)  # type: ignore

    # ----------------------
    # Identify and store which atoms, bonds, and rings we'll be highlighting
    highlightatoms = defaultdict(list)
    highlightbonds = defaultdict(list)
    atomrads = {}
    widthmults = {}

    rings = []
    for i, lbl in enumerate(lbls):
        color = colors[i % len(colors)]  # modulus to obtain closest colours.
        try:
            rquery = row[lbl]
        except (KeyError):
            continue
        Chem.GetSSSR(rquery)
        rinfo = rquery.GetRingInfo()
        for at in rquery.GetAtoms():
            if at.HasProp(sourceIdxProperty):
                origIdx = oldNewAtomMap[at.GetIntProp(sourceIdxProperty)]
                highlightatoms[origIdx].append(color)
                atomrads[origIdx] = 0.4
        if fillRings:
            for aring in rinfo.AtomRings():
                tring = []
                allFound = True
                for aid in aring:
                    at = rquery.GetAtomWithIdx(aid)
                    if not at.HasProp(sourceIdxProperty):
                        allFound = False
                        break
                    tring.append(oldNewAtomMap[at.GetIntProp(sourceIdxProperty)])
                if allFound:
                    rings.append((tring, color))
        for qbnd in rquery.GetBonds():
            batom = qbnd.GetBeginAtom()
            eatom = qbnd.GetEndAtom()
            if batom.HasProp(sourceIdxProperty) and eatom.HasProp(sourceIdxProperty):
                origBnd = tmol.GetBondBetweenAtoms(
                    oldNewAtomMap[batom.GetIntProp(sourceIdxProperty)],
                    oldNewAtomMap[eatom.GetIntProp(sourceIdxProperty)],
                )
                bndIdx = origBnd.GetIdx()
                highlightbonds[bndIdx].append(color)
                widthmults[bndIdx] = 2

    d2d = rdMolDraw2D.MolDraw2DCairo(width, height)
    dos = d2d.drawOptions()
    dos.useBWAtomPalette()

    # ----------------------
    # if we are filling rings, go ahead and do that first so that we draw
    # the molecule on top of the filled rings
    if fillRings and rings:
        # a hack to set the molecule scale
        d2d.DrawMoleculeWithHighlights(
            tmol, legend, dict(highlightatoms), dict(highlightbonds), atomrads, widthmults
        )
        d2d.ClearDrawing()
        conf = tmol.GetConformer()
        for (aring, color) in rings:
            ps = []
            for aidx in aring:
                pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
                ps.append(pos)
            d2d.SetFillPolys(True)
            d2d.SetColour(color)
            d2d.DrawPolygon(ps)
        dos.clearBackground = False

    # ----------------------
    # now draw the molecule, with highlights:
    d2d.DrawMoleculeWithHighlights(
        tmol, legend, dict(highlightatoms), dict(highlightbonds), atomrads, widthmults
    )
    d2d.FinishDrawing()
    png = d2d.GetDrawingText()
    return png


def draw_multiple(
    ms: Union[str, Chem.Mol],
    groups: List,
    qcore: Union[str, Chem.Mol],
    lbls: Tuple[str],
    legends=None,
    molsPerRow=4,
    subImageSize=(250, 200),
) -> pilImage.Image:
    """Manually combine images of highlighted and unhighlighted molecules.

    Parameters
    ----------
    ms : Union[str, Chem.Mol]
        RDKit Mols
    groups : List
        List of all decomposed R-groups.
    qcore : Union[str, Chem.Mol]
        Core of molecule R-groups are attached to.
    lbls : Tuple[str]
        R-group labels to display.
    legends : str, optional
        Quantity to display as legend on molecule plots, by default None
    molsPerRow : int, optional
        NUmber of molecules to display in  a given row, by default 4
    subImageSize : tuple, optional
        Image height x width to display in grid, by default (250,200)

    Returns
    -------
    pilImage.Image
        _description_
    """
    nRows = len(ms) // molsPerRow
    if len(ms) % molsPerRow:
        nRows += 1
    nCols = molsPerRow
    imgSize = (subImageSize[0] * nCols, subImageSize[1] * nRows)
    res = pilImage.new('RGB', imgSize)

    for i, m in enumerate(ms):
        col = i % molsPerRow
        row = i // molsPerRow
        if legends:
            legend = legends[i]
        else:
            legend = ''
        png = highlight_rgroups(
            m,
            groups[i],
            qcore,
            lbls=lbls,
            legend=legend,
            width=subImageSize[0],
            height=subImageSize[1],
        )
        bio = BytesIO(png)
        img = pilImage.open(bio)
        res.paste(img, box=(col * subImageSize[0], row * subImageSize[1]))
    return res

def dataframe_to_sdf(
    df: pd.DataFrame,
    mol_col_name: str,
    suffix: str,
    base_dir: str = ".",
    name_col_name: str = "prepare_mol_inchikey",
    overwrite: bool = False,
    props: List[str] = [],
    v3000: bool = True,
) -> List[str]:
    """Convert a dataframe to a sdf file.
    return str

    Parameters
    ----------

    df: pd.DataFrame
        List of s3 paths
    mol_col_name: str
        column where mol rdkit is
        (ex: conformer_mol, pose_mol, relaxation_mol)
    suffix: str
        add your suffix to output filename
    name_col_name: str
        column name to use for _Name prop set up,
        usualluy prepare_mol_inchikey
    props: List[str]
        Columns you would like to add to sdf props.
    v3000: bool
        set v3000 format
    Returns
        output_path
    -------
    """
    now = datetime.now()
    prefix = now.strftime("%Y_%m_%d")

    mandatory_props = [
        "standard_inchikey",
        "prepare_mol_inchikey",
        "run_id",
        "s3_bucket",
        "job_timestamp",
    ]
    optional_props = [
        "source",
        "idName",
        "tautomer_distribution",
        "protonation_state_distribution",
        "net_molecular_charge",
        "conformer_hash",
        "conformer_inchikey",
        "conformer_tors_score",
        "conformer_energy",
        "pose_hash",
        "pose_inchikey",
        "protein_id",
        "docking_tors_score",
        "tleap_config_s3_path",
        "protein_pdb_s3_path",
        "relaxation_inchikey",
        "relaxation_tors_score",
        "relaxation_e_inter",
        "relaxation_e_lj",
        "relaxation_e_elec",
        "relaxation_e_tot_complex",
        "relaxation_e_tot_protein",
        "relaxation_e_tot_ligand_relaxed_alone",
        "relaxation_e_tot_ligand_relaxed_alone",
        "complex_prmtop_s3_path",
        "complex_relaxed_rst7_s3_path",
        "complex_relaxed_pdb_s3_path",
        "_score",
        "strain",
        "_score_with_ml_correction",
        "dg_sol_PL",
        "dg_sol_P",
        "dg_sol_L",
    ]

    df = global_utils.convert_json_to_mol_in_df(df, mol_col_name)

    optional_props = list(set(props + optional_props))

    if name_col_name not in df.columns:
        raise ValueError(f"Missing name_col_name {name_col_name} column in your dataframe.")

    for mandatory_prop in mandatory_props:
        if mandatory_prop in optional_props:
            optional_props.remove(mandatory_prop)
        if mandatory_prop not in df.columns:
            raise ValueError(f"Missing {mandatory_prop} column in your dataframe.")

    if "protein_id" in df.columns:
        protein_ids = list(set(df["protein_id"]))
    else:
        protein_ids = ["noProtein"]

    s3_buckets = list(set(df["s3_bucket"]))
    if len(s3_buckets) > 1:
        raise ValueError("Multiple s3_bucket in your dataframe")
    s3_bucket = s3_buckets[0]

    missing_optional_props = []
    for optional_prop in optional_props:
        if optional_prop not in df.columns.to_list():
            missing_optional_props.append(optional_prop)
            if optional_prop in props:
                logger.warning(f"{optional_prop} not in df.")

    output_paths = []
    for protein_id in protein_ids:
        if protein_id != "noProtein":
            df_protein = df[df["protein_id"] == protein_id].reset_index(drop=True)
        else:
            df_protein = df
        outpath = f"{base_dir}/{prefix}_{s3_bucket}_{protein_id}_{suffix}.sdf"
        output_paths.append(outpath)
        if os.path.exists(outpath) and not overwrite:
            logger.error(f"{outpath} already exists.")
            return []
        writer = Chem.SDWriter(outpath)
        if v3000:
            writer.SetForceV3000(True)

        fails = []
        for i, row in tqdm(df_protein.iterrows()):
            try:
                for p in mandatory_props:
                    row[mol_col_name].SetProp(p, str(row[p]))
                for p in optional_props:
                    if p not in missing_optional_props:
                        row[mol_col_name].SetProp(p, str(row[p]))
                row[mol_col_name].SetProp("_Name", str(row[name_col_name]))
                writer.write(row[mol_col_name])
            except Exception as e:
                fails.append((i, e))
        if len(fails) > 0:
            logger.error(f"Protein {protein_id}: fail to convert {len(fails)} mol(s).")
        writer.close()
        logger.info(f"Sdf file saved to {outpath}")
    return output_paths