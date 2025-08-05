from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from utils import drawing

import collections
import logging
import re
import typing
import warnings
from typing import (
    Callable,
    Dict,
    Generator,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def draw_smarts_mol(mol: Chem.Mol):
    """
    Draw the smarts as a PIL image. It is more permisive than `draw_moeity_image` to support weird
    kekulization in SMARTS Chem.Mol

    Parameters
    ----------
    mol : Chem.Mol
        Chem.Mol object corresponding to a SMARTS
    """
    drawOptions = rdMolDraw2D.MolDrawOptions()
    drawOptions.prepareMolsBeforeDrawing = False
    try:
        mol_draw = rdMolDraw2D.PrepareMolForDrawing(mol)
    except Chem.KekulizeException:
        mol_draw = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
        Chem.SanitizeMol(mol_draw, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    return mol_draw


def draw_moeity_image(mol: Chem.Mol):
    """
    Draw the moiety as a PIL image

    Parameters
    ----------
    mol : Chem.Mol
        Chem.Mol object corresponding to the moiety
    """
    return Draw.MolToImage(mol)


def draw_molecule(molecule: str):
    """
    Draw a molecule as a PIL image

    Parameters
    ----------
    molecule : str
        Chem.Mol object corresponding to the molecules
    """
    return Draw.MolToImage(Chem.MolFromSmiles(molecule))


def draw_molecule_aligned_on_moiety(molecule: str, moiety: Chem.Mol) -> Image.Image:
    """
    Draw a molecule as a PIL image, aligned with the moiety structure. Moiety must be a
    mol resulting from the function drawing.mol_2d(SMILES)

    Parameters
    ----------
    molecule : str
    moiety : Chem.Mol
    """
    return drawing.draw_2d_png(Chem.MolFromSmiles(molecule), aligned_on=moiety)


def draw_molecule_aligned_on_moiety_svg(molecule: str, moiety: Chem.Mol) -> str:
    """
    Draw a molecule as a svg image, aligned with the moiety structure. Moiety must be a
    mol resulting from the function drawing.mol_2d(SMILES)

    Parameters
    ----------
    molecule : str
    moiety : Chem.Mol
    """
    return drawing.draw_2d_svg(
        Chem.MolFromSmiles(molecule), aligned_on=moiety, background_color=(0, 0, 0, 0)
    )


def draw_molecule_svg(
    molecule: str,
) -> str:
    """
    Draw a molecule as a svg image.

    Parameters
    ----------
    molecule : str
    """
    return drawing.draw_2d_svg(Chem.MolFromSmiles(molecule), background_color=(0, 0, 0, 0))

def is_stereo_attachment_point(mol: Chem.Mol, attach_idx: int) -> bool:
    """
    On a molecule without explicit Hs, checks if the chosen AP
    will become a stereocenter after addition.

    Parameters
    ----------
    mol : Chem.Mol
        the mol to decorate
    idx : int
        index of the

    Returns
    -------
    bool
        [description]
    """
    atom = mol.GetAtomWithIdx(attach_idx)
    at_num = atom.GetAtomicNum()
    imp_val = atom.GetImplicitValence()

    return at_num == 6 and imp_val == 2


def get_attachment_mask(mol: Chem.Mol) -> np.ndarray:
    """
    Given a mol with N atoms, returns an binary mask of size N
    indicating the atoms suitable for an attachment.
    This function does not guarantee that the atom mutation
    will be possible because it can still break the aromaticity
    in double rings, but it limits the search space.

    Parameters
    ----------
    mol : Chem.Mol
        mol object without explicit Hs

    Returns
    -------
    np.ndarray
        binary mask of size mol.GetNumAtoms()
    """

    mask = np.zeros(mol.GetNumAtoms())
    for i, atom in enumerate(mol.GetAtoms()):
        atom.UpdatePropertyCache()

        at_num = atom.GetAtomicNum()
        imp_val = atom.GetImplicitValence()

        if imp_val > 0 and at_num > 1:  # not H, not dummy
            mask[i] = 1

    return mask


def get_cuttable_bonds_mask(mol: Chem.Mol) -> np.ndarray:
    """
    Given a mol with N_b bonds, returns an binary mask of size N_b
    indicating the bonds that can be cut.

    Parameters
    ----------
    mol : Chem.Mol
        mol object

    Returns
    -------
    np.ndarray
        binary mask of size len(mol.GetBonds())
    """

    bonds = mol.GetBonds()
    mask = np.zeros(len(bonds))

    for i, b in enumerate(bonds):
        if not b.GetIsAromatic() and not b.IsInRing():
            mask[i] = 1

    return mask


def get_bond_atom_indices(mol, bond_idx: int) -> Tuple[int, int]:
    bond = mol.GetBondWithIdx(int(bond_idx))  # need to cast np.int64 to int
    return bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()


def add_attachment(mol: Chem.Mol, attach_idx: int, stereo: str = None) -> Chem.Mol:
    """
    Adds a dummy atom marking attachment possibility on the
    atom of index attach_idx in the input molecule

    Parameters
    ----------
    mol : Chem.Mol
    attach_idx : int
        Index of atom to be marked as attachment point
    stereo : str
        whether to add the attachment * clockwise or anticlockwise.poetry

    Returns
    -------
    Chem.Mol
        Input molecule with a * attached to the attachment atom
    """

    emol = Chem.RWMol(mol)  # Create new editable mol
    new_atom = Chem.Atom('*')
    w_idx = emol.AddAtom(new_atom)  # w_idx is the index of the wildcard atom

    emol.AddBond(w_idx, int(attach_idx), Chem.rdchem.BondType.SINGLE)

    atom = emol.GetAtomWithIdx(int(attach_idx))
    if stereo is None:
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    elif stereo == 'CW':
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
    elif stereo == 'CCW':
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
    else:
        raise ValueError(f'Stereo {stereo} tag not implemented')

    return emol.GetMol()


def add_atoms_with_attachment(emol: Chem.RWMol, mol: Chem.Mol) -> Tuple[Dict[int, int], int]:
    """
    Updates the provided editable mol inplace, by adding all atoms from the provided mol to it.
    Returns a mapping from atom numbers in mol to their id in the editable mol,
    and the index of the attachment point in the editable mol

    Parameters
    ----------
    emol : Chem.RWMol
        Rdkit editable mol to be completed
    mol : Chem.Mol
        mol to be "added" to the editable mol

    Returns
    -------
    Dict, int
        - Dictionary mapping from atom ids in mol to ids in editable mol
        - index of the attachment point in editable mol
    """

    # Mapping from original mol to new mol
    a_map = {}

    attach_idx = -1
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        atom_chiraltag = atom.GetChiralTag()

        if atom_symbol == '*':
            neighbors = atom.GetNeighbors()
            assert len(neighbors) == 1  # wildcard atom should only have 1 neighbor
            attach_idx = neighbors[0].GetIdx()
            continue
        atom_fc = atom.GetFormalCharge()

        new_atom = Chem.Atom(atom_symbol)
        new_atom.SetFormalCharge(atom_fc)
        new_atom.SetChiralTag(atom_chiraltag)
        if atom_symbol != 'C':
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())

        new_atom_idx = emol.AddAtom(new_atom)
        a_map[atom_idx] = new_atom_idx
    assert attach_idx != -1

    return a_map, attach_idx


def add_bonds(emol: Chem.RWMol, mol: Chem.Mol, a_map: Dict[int, int]):
    """
    Adds bonds on editable mol

    Parameters
    ----------
    emol : Chem.RWMol
        [description]
    mol : Chem.Mol
        [description]
    a_map : [type]
        [description]
    """

    for bond in mol.GetBonds():
        # Ignore the wildcard atoms
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        a1_idx, a2_idx = a1.GetIdx(), a2.GetIdx()
        if a1.GetSymbol() == '*' or a2.GetSymbol() == '*':
            continue
        bond_type = bond.GetBondType()
        emol.AddBond(a_map[a1_idx], a_map[a2_idx], bond_type)


def as_mol(mol: Union[str, Chem.Mol]) -> Chem.Mol:
    """
    Convert an object to a RdKit molecule

    This function supports SMILES and molecules encoded as JSON. Molecule
    object are returned as-is. A useful error is raised if conversion fails,
    instead of returning 'None', which laters triggers unhelpful exception
    like AttributeError.

    Parameters
    ----------
    mol : Union[str, Chem.Mol]
        SMILES, JSON or Chem.Mol object to convert

    Returns
    -------
    Chem.Mol
        The passed object converted to rdkit.Chem.Mol type, if possible

    Raises
    ------
    ValueError
        The conversion failed
    """
    msg = ""
    if isinstance(mol, str):
        # Quick differenciation between JSON and SMILES
        if "{" in mol:
            mols = Chem.JSONToMols(mol)
            if len(mols) > 1:
                msg = "JSON contains more than 1 molecule"
            elif len(mols) < 1:
                msg = "JSON contains no molecules"
            else:
                return mols[0]
        else:
            ret = Chem.MolFromSmiles(mol)
            if ret is not None:
                return ret
    elif isinstance(mol, Chem.Mol):
        return mol
    sep = ": " * bool(msg)
    raise ValueError(
        f"Could not convert {mol} of type {type(mol).__name__} to rdkit.Chem.Mol object{sep}{msg}"
    )


def flatten_mol(mol: Union[str, Chem.Mol]) -> Chem.Mol:
    """
    Converts a Molecule object to SMILES and back to remove 3D coordinates

    Parameters
    ----------
    mol : Union[str, Chem.Mol]
        SMILES, JSON or Mol to flatten

    Returns
    -------
    Chem.Mol
        Flattened version of ``mol``
    """
    mol = Chem.Mol(as_mol(mol))
    mol.RemoveAllConformers()
    return mol


def with_canonical_indexing(mol: Union[str, Chem.Mol]) -> Chem.Mol:
    """
    Renumber the atoms of a molecule to rdkit canonical numbering

    This is the numbering obtain with rdkit.Chem.CaninicalRankAtoms()

    Parameters
    ----------
    smiles_or_mol : Union[str, Chem.Mol]
        SMILES, JSON or Mol object to renumber

    Returns
    -------
        A copy if the input molecule, where the atoms are re-numbered to the rdkit caninical order
    """
    mol = Chem.Mol(as_mol(mol))

    # I) CLEAN atom annotations
    #    Chem.CanonicalRankAtoms() takes the annotations into account, even with
    #    and includeChirality=False, includeIsotopes=False
    #    We do not want to depend on AtomMapNum, so we reorder a copy of the molecule
    #    without the annotations
    mol_clean = Chem.Mol(mol)
    for atom in mol_clean.GetAtoms():
        atom.SetAtomMapNum(0)

    # II) Compute the new order on the cleaned molecules, ignoring Chirality and Isotopes
    #   from https://www.rdkit.org/docs/Cookbook.html#reorder-atoms
    #   explanation of the expression from https://gist.github.com/ptosco/36574d7f025a932bc1b8db221903a8d2
    canonic_idx_to_orig_idx = [
        (canonic, orig)
        for orig, canonic in enumerate(
            Chem.CanonicalRankAtoms(mol_clean, includeChirality=False, includeIsotopes=False)
        )
    ]
    orig_idx_ordered_by_canon_idx = tuple(zip(*sorted(canonic_idx_to_orig_idx)))[1]

    # III) Reorder the orginal molecule. The AtomIds have nto changes
    # when we copied the molecule, and we didn't change them afterwrads
    return Chem.RenumberAtoms(mol, orig_idx_ordered_by_canon_idx)


def with_canonical_numbering(mol: Union[str, Chem.Mol]) -> Chem.Mol:
    """See `with_caononical_indexing`"""
    warnings.warn(
        "with_canonical_numbering() has been renamed with_canonical_indexing() and is deprecated",
        DeprecationWarning,
    )
    return with_canonical_indexing(mol)


def with_atomid(mol: Union[str, Chem.Mol], *, offset: int = 1) -> Chem.Mol:
    """
    Adds the atom IDs as annotations on the molecule's atoms

    Parameters
    ----------
    smiles_or_mol : Chem.Mol
        The SMILES, JSON or molecule to annotate
    offset : int, optional, default=1
        Starting point of the atom ID, by default 0. This can be useful to start at 1 instead of
        0, because rdkit uses 0 as the 'no value', so the first atom misses an annotation.

    Returns
    -------
    Chem.Mol
        Input molecule with additional annotation on atoms. Molecule objects are edited in-place.
    """
    mol = Chem.Mol(as_mol(mol))
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + offset)
    return mol


def with_canonical_atomid(mol: Chem.Mol, *, offset: int = 1) -> Chem.Mol:
    """
    Adds the canonical atom IDs as annotations on the molecule's atoms

    This is a shortcut to calling `with_canonical_indexing` and `with_atomid`

    Parameters
    ----------
    smiles_or_mol : Chem.Mol
        The SMILES, JSON or molecule to annotate
    offset : int, optional, default=1
        Starting point of the atom ID, by default 0. This can be useful to start at 1 instead of
        0, because rdkit uses 0 as the 'no value', so the first atom misses an annotation.

    Returns
    -------
    Chem.Mol
        Input molecule with additional annotation on atoms. Molecule objects are edited in-place.
    """
    return with_atomid(with_canonical_indexing(mol), offset=offset)


def get_substitutions(
    core: Union[str, Chem.Mol], smiles_or_mol: Union[str, Chem.Mol]
) -> Dict[Union[int, Tuple[int, ...]], List[str]]:
    """
    Finds the substitutions on a core

    Gets all the fragments/sidechains/substitutions attached on the provided core.

    Parameters
    ----------
    core : Union[str, Chem.Mol]
        SMILES or mol of the core to look for substitutions
    smiles_or_mol : Union[str, Chem.Mol]
        SMILES or mol of the molecules to extract substitutions from

    Returns
    -------
    Dict[Union[int, Tuple[int, ...]], List[str]]
        A mapping from core atoms canonical ID (as per `with_canonical_numbering`) to the list
        of subsitutions attached to that atom, as a SMILES with a dummy atom.

        Side-chains attached to several core atom are mapped from a tuple of atom IDs£.

        Atom ID starts at 1, instead of 0 as in rdkit internals, for compat with `set_substitutions`.
    """
    core_mol = with_canonical_indexing(core)

    mol = as_mol(smiles_or_mol)
    matches = mol.GetSubstructMatches(core_mol)
    if len(matches) < 1:
        raise ValueError(f"{Chem.MolToSmiles(mol)} doesn't match {Chem.MolToSmiles(core_mol)}")
    elif len(matches) > 1:
        raise ValueError(
            f"Multiple matches of {Chem.MolToSmiles(core_mol)} in {Chem.MolToSmiles(mol)}"
        )
    else:
        fragment_mol = Chem.ReplaceCore(
            mol=mol, core=core_mol, matches=mol.GetSubstructMatch(core_mol), labelByIndex=True
        )
        fragments: List[str] = Chem.MolToSmiles(fragment_mol).split(".")
        substitutions: Dict[Union[int, Tuple[int, ...]], List[str]] = {}
        is_attachement = re.compile(r"^(?P<ap>\d*)\*$").match
        for fragment in fragments:
            if fragment:
                attachement_atom_ids = []
                chain_parts = []
                index = 0
                for token, id_ in tokenize_smiles(fragment):
                    # Note: ReplaceCore above yield SMILES with things such as * or [*3]
                    #       as there is no ':', _tokenize_smiles above does not put the number
                    #       in the id_ field. We parse those ourselves
                    if match := is_attachement(token):
                        attachement_atom_ids.append(int(match.group("ap") or 0) + 1)
                        chain_parts.append(f"[*:{index}]")
                        index += 1
                    else:
                        # Preserve the annotations and brackets, that's useful to map atom id
                        if id_ is not None:
                            chain_parts.append(f"[{token}:{id_}]")
                        elif len(token) > 1:
                            chain_parts.append(f"[{token}]")
                        else:
                            chain_parts.append(token)
                chain = "".join(chain_parts)
                attachement: Union[int, Tuple[int, ...]]
                if len(attachement_atom_ids) == 1:
                    attachement = attachement_atom_ids[0]
                    chain = chain.replace("[*:0]", "*")
                else:
                    attachement = tuple(attachement_atom_ids)
                substitutions.setdefault(attachement, []).append(chain)
        return substitutions


def get_substitution_indexes(
    core: Union[str, Chem.Mol], smiles_or_mol: Union[str, Chem.Mol]
) -> Dict[Union[int, Tuple[int, ...]], List[List[int]]]:
    """
    Finds the indexes of substitutions on a core

    Gets all the fragments/sidechains/substitutions attached on the provided core, in
    the form of their canonical atom IDs.

    As in all functions in mol_utils, canonical atom IDs start at 1.

    Parameters
    ----------
    core : Union[str, Chem.Mol]
        SMILES or mol of the core to look for substitutions
    smiles_or_mol : Union[str, Chem.Mol]
        SMILES or mol of the molecules to extract substitutions from

    Returns
    -------
    Dict[Union[int, Tuple[int, ...]], List[str]]
        A mapping from core atoms canonical ID (as per `with_canonical_numbering`), or a tuple
        thereof when the substitution is attached to several core atoms, to the list
        of atom IDs in that subsitutions.

        This is very similar to `get_substitutions`, and actually use that function internally,
        but returns atom IDs instead of the chain's SMILES representation.

        This is useful e.g. to highligh the side-chains when plotting a molecule
    """
    mol = with_canonical_atomid(smiles_or_mol)
    substitutions = get_substitutions(core=core, smiles_or_mol=mol)
    return {
        key: [
            [id_ for token, id_ in tokenize_smiles(chain) if token != "*" and id_ is not None]
            for chain in chains
        ]
        for key, chains in substitutions.items()
    }


def tokenize_smiles(
    smiles: Union[str, Chem.Mol]
) -> Generator[Tuple[str, Optional[int]], None, None]:
    """
    Splits an (annotated smiles) into components

    The SMILES is split into component. A component is either all the entities in
    square brackets, or a single character. If a token is annotated, by the
    SetAtomMapNum from RdKit, the mapped number is return independently of the
    token.

    This is a rather internal function, but it can be useful to retrieve the
    annotations on a SMILES, so it is exposed.

    Parameters
    ----------
    smiles : str, Chem.Mol
        The SMILES to tokenize. If a Chem.Mol object, it is converted to SMILES

    Returns
    -------
    Generator[Tuple[str, Optional[int]], None, None]
        A generator (something you can iterate on in a for-loop) that yields 2-tuples
        of (token, annotation). The annotation may be None, for component that
        do not have annotations

    Example
    -------
        >>> list(tokenize_smiles("c1cc[c:3]nc1"))
        >>> [('c', None),
             ('1', None),
             ('c', None),
             ('c', None),
             ('c', 3),
             ('n', None),
             ('c', None),
             ('1', None)]
    """
    if isinstance(smiles, Chem.Mol):
        smiles = Chem.MolToSmiles(smiles)
    i = 0
    while i < len(smiles):
        if smiles[i] == "[":
            end = smiles.find("]", i)
            if end < 0:
                raise ValueError(f"Unmatched '[' in smiles at index {i}")
            j = smiles.rfind(":", i, end + 1)
            if j > 0:
                yield (smiles[i + 1 : j], int(smiles[j + 1 : end]))
            else:
                yield (smiles[i + 1 : end], None)
            i = end + 1
        else:
            yield (smiles[i], None)
            i += 1


def add_dummy_atoms(mol: Union[str, Chem.Mol], atom_ids: Iterable[int]) -> Chem.Mol:
    """
    Add dummy atoms to a molecule to prepare modifications

    The atom IDs used are canonical atom ID (i.e. canonical atom indexes offset by 1).
    This is the same interface as all mol_utils function, and is because we use
    ``SetAtomMapNum() ``internally, for which 0 is the "empty" value.

    This function removes hydrogens from the molecule, otherwise arity problem ensue.

    Parameters
    ----------
    mol : Union[str, Chem.Mol]
        A molecule, in any format accepted by `as_mol`
    atom_ids : Iterable[int]
        An iterable of canonical atom ids to which to attach dummy atoms. You can
        view the canonical atom IDs with `with_canonical_atomid`.
        If you plan to attach multiple time to the same atom, you should pass an
        iterable containing the same atom ID multiple time, one for each attachement
        points. The resulting dummy atom are numbered by adding 100 times the number
        of replicates (that is XX, 1XX, 2XX, etc...).

    Returns
    -------
    Chem.Mol
        A new molecule with additional dummy atom, with an AtomMapNum equal to the
        (old) index of the atom they are bound to.
    """
    atom_counts = collections.Counter(atom_ids)
    rwmol = AllChem.RWMol(with_canonical_indexing(mol))
    rwmol.BeginBatchEdit()
    h_to_remove: List[AllChem.Atom] = []
    for atom_id, count in atom_counts.items():
        # Offset of 1, because we use AtomMapNum, and 0 is the 'empty' value
        # This is the same in all mol_utils interfaces
        atom = rwmol.GetAtomWithIdx(atom_id - 1)
        # Remove hydrogens on the target atom
        h_to_remove.extend(atom for atom in atom.GetNeighbors() if atom.GetAtomicNum() == 1)
        for replicate in range(count):
            # Add a dummy atom
            dummy_atom = AllChem.Atom("*")
            dummy_atom.SetAtomMapNum(100 * replicate + atom_id)
            dummy_id = rwmol.AddAtom(dummy_atom)
            # Don't forget that the true internal atom_id is offset
            rwmol.AddBond(atom_id - 1, dummy_id)
    for atom in h_to_remove:
        rwmol.RemoveAtom(atom.GetIdx())
    rwmol.CommitBatchEdit()
    new_mol = rwmol.GetMol()
    new_mol.UpdatePropertyCache()
    Chem.SanitizeMol(new_mol)
    return with_canonical_indexing(new_mol)


def _make_get_count() -> Callable[[Hashable], int]:
    counter: collections.Counter = collections.Counter()

    def get(key: Hashable) -> int:
        count = counter[key]
        counter[key] = count + 1
        return count

    return get


def set_substitutions(
    mol_or_smiles: Union[str, Chem.Mol],
    substitutions: Mapping[Union[int, Tuple[int, ...]], Union[str, List[str]]],
) -> Chem.Mol:
    """
    Set the substitutions on a core, or attach fragments to a molecule

    Parameters
    ----------
    mol_or_smiles : Union[str, Chem.Mol]
        The core or base molecule to attach substitutions on
    substitutions : Dict[int, List[str]]
        Substitutions to attach. Should map the canonical atom ID (as per
        `with_canonical_numbering`) of the molecule to the list of chains to attach
        on that position. Side-chain must have a dummy '*' atom to indicate where to
        attach them (it is included in the return value of `get_subsitutions`.)
        Atom ID starts at 1, instead of 0 like the internals of rdkit,
        because attaching at index 0 is not possible with the rdkit uitlities this
        function uses.

    Returns
    -------
    Chem.Mol
        The core with the attached substitutions
    """
    # Prepare a SMILES of the core with dummy atoms numbered
    # by the atom_id of the atom they are on
    used_atom_ids: List[int] = []
    for key, chains in substitutions.items():
        length = 1 if isinstance(chains, str) else len(chains)
        if isinstance(key, tuple):
            # We need duplicate indices for each attachement
            used_atom_ids.extend(key * length)
        else:
            used_atom_ids.extend((key for _ in range(length)))
    parts = [Chem.MolToSmiles(add_dummy_atoms(mol_or_smiles, used_atom_ids))]

    # Re-map the chains to actual atom ID instead of indexes
    get_count = _make_get_count()
    for key, chains in substitutions.items():
        if isinstance(chains, (str)):
            chains = [chains]
        if isinstance(key, int):
            # Only one attachement point
            for chain in chains:
                if "*" not in chain:
                    raise ValueError(
                        f"No dummy atom '*' in chain '{chain}' at attachement id {key}"
                    )
                dummy = 100 * get_count(key) + key
                parts.append(chain.replace("*", f"[*:{dummy}]"))
        else:
            for chain in chains:
                chain_parts: List[str] = []
                for token, id_ in tokenize_smiles(chain):
                    if token == "*":
                        dummy = 100 * get_count(key[id_ or 0]) + key[id_ or 0]
                        token = f"[*:{dummy}]"
                        if chain_parts and chain_parts[-1][-1] != "(":
                            token = "(" + token + ")"
                    chain_parts.append(token)
                new_chain = "".join(chain_parts)
                parts.append(new_chain)
    mol = Chem.molzip(as_mol(".".join(parts)))
    Chem.SanitizeMol(mol)
    return with_canonical_indexing(mol)


def replace_atoms(
    mol: Union[str, Chem.Mol],
    replacements: Dict[int, Union[int, str]],
    *,
    map_index_from_substruct: Union[str, Chem.Mol] = None,
    add_hs: bool = False,
) -> Chem.Mol:
    """
    Replace atoms in a molecule, leaving 3D coordinates unchanged.

    The replaced atoms keeps their exact same coordinates. This might yield a molecule that
    has an improbable conformation, because e.g. bond length, torsion angles etc... are not
    physically possible or very high energy. It is recommended to relax the molecule before
    using it.

    This function preserves aromaticity of the replaced atom, because changing an aromatic
    cyle to an aliphatic cycle requires more than simple atomic replacements.

    Replacements are specified from canonical atom ids, as computed by
    `with_canonical_numbering`. Use `with_atomid` to show the new index on your input mol.
    For consistency with `get_substitutions` and `set_substitutions`, the indexes used by
    this function starts at 1, while the RdKit internal index start at 0. This function will
    perform the necessary decrement itself.

    Parameters
    ----------
    mol : Union[str, Chem.Mol]
        SMILES, JSON or Mol object to change atoms from
    replacements : Dict[int, Union[int, str]]
        Mapping from canonical atom indexes to the atomic number or atom symbol to replace
        the atom with. Atom symbol can be atomic number or the letter representation.
    map_index_from_substruct : Union[str, Chem.Mol], optional, default=None
        SMILES, JSON or Mol object to use to remap indexes in `replacements`, by default None.
        When this is specified, the index in `replacements` should be canonical index of this
        molecule, instead of ``mol``. A substructure search is used to map index from this
        substructure to index of the actual mol.
        This makes it possible to replace atoms from many molecules which share a common parts,
        since such molecule do not have a coherent canonical atom indexing.
    add_hs : bool, optional, default=False
        Add hydrogen after atom modification. This function removes the hydrogens bond to replaced
        atoms because their valence may change. This option adds back the hydrogen, but their 3D
        coordinates will be bad. Furthermore, most RdKit function assume there are no hydrogens
        are in the molecule, so this is false by default.

    Returns
    -------
    Chem.Mol
        A copy of the input mol object, with edited atoms, and canonical atom indexing
    """
    mol = with_canonical_indexing(mol)
    # remove 1 from all IDs, to keep indexing consistent wit other functions
    replacements = {atom_id - 1: atom_symbol for atom_id, atom_symbol in replacements.items()}
    # Map indexes
    if map_index_from_substruct is not None:
        substruct = with_canonical_indexing(map_index_from_substruct)
        matches = mol.GetSubstructMatches(substruct)
        if len(matches) < 1:
            raise ValueError(
                f"No substructure matching {Chem.MolToSmiles(substruct)} in {Chem.MolToSmiles(mol)}"
            )
        elif len(matches) > 1:
            raise ValueError(
                f"Multiple subtructure matching {Chem.MolToSmiles(substruct)} in {Chem.MolToSmiles(mol)}"
                ", can't define an index mapping"
            )
        match = mol.GetSubstructMatch(substruct)
        # get SubstructMatch is documented to return atom IDs in the order
        # of the atom ids of the substruct
        idx_map = {substruct_id: mol_id for substruct_id, mol_id in enumerate(match)}
        replacements = {idx_map[idx]: atom_symbol for idx, atom_symbol in replacements.items()}

    # Perform replacements
    # Create a mapping of atom indexes
    rwmol = AllChem.RWMol(mol)
    rwmol.BeginBatchEdit()
    # CHanging atom most likely changes the valence, so we'll remove all Hs
    # bond to modified atoms, then re-add them.
    h_to_remove: List[AllChem.Atom] = []
    for atom_id, atom_symbol in replacements.items():
        atom = rwmol.GetAtomWithIdx(atom_id)
        h_to_remove.extend(atom for atom in atom.GetNeighbors() if atom.GetAtomicNum() == 1)
        new_atom = AllChem.Atom(atom_symbol)
        # Preserve aromaticity -- RdKit loudly complains otherwise
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        rwmol.ReplaceAtom(atom_id, new_atom, preserveProps=True)
    for atom in h_to_remove:
        rwmol.RemoveAtom(atom.GetIdx())
    rwmol.CommitBatchEdit()
    new_mol = rwmol.GetMol()
    new_mol.UpdatePropertyCache()
    Chem.SanitizeMol(new_mol)
    if add_hs:
        new_mol = Chem.AddHs(new_mol, addCoords=True)
    return with_canonical_indexing(new_mol)


def _get_minimal_core_atomids(mol: Chem.Mol, atom_ids: List[int]) -> Set[int]:
    """
    Selects the atoms that are part of the minimal scaffold containing the atom_ids

    The minimal scaffold is the subpart of the molecule that contains the passed atom_ids
    and preserves rings and aromaticity.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to select atoms from
    atom_ids : List[int]
        The IDs of the atom to start the scaffold from

    Returns
    -------
    Set[int]
        A set of atom IDs that are part of the scaffold

    Raises
    ------
    ValueError
        _description_
    """
    import networkx as nx

    selected_atoms = set()
    if len(atom_ids) == 0:
        raise ValueError("Cannot extract scaffold starting with no atoms")
    elif len(atom_ids) == 1:
        selected_atoms.add(atom_ids[0])
    else:
        # Add the atoms necessary to connect the contacts
        graph = nx.Graph()
        for bond in mol.GetBonds():
            graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        paths = typing.cast(
            Dict[int, List[int]], nx.single_source_dijkstra_path(graph, atom_ids[0])
        )
        for atom_id in atom_ids[1:]:
            selected_atoms.update(paths[atom_id])
    # Complete rings
    rings: List[Set[int]] = list(map(set, mol.GetRingInfo().AtomRings()))
    while True:
        modified = False
        for ring in rings:
            if ring & selected_atoms and ring - selected_atoms:
                modified = True
                selected_atoms.update(ring)
        if not modified:
            break

    # Add atoms bounded to ring atoms that affect aromaticity
    # See https://www.rdkit.org/docs/RDKit_Book.html
    for atom_id in list(selected_atoms):
        atom = mol.GetAtomWithIdx(atom_id)
        if atom.GetIsAromatic() and atom.GetAtomicNum() == 6:
            # Carbon are sensitive to exocyclic double-bounded atoms, add them
            for neighbor in atom.GetNeighbors():
                if (
                    mol.GetBondBetweenAtoms(atom_id, neighbor.GetIdx()).GetBondType()
                    == Chem.BondType.DOUBLE
                ):
                    selected_atoms.add(neighbor.GetIdx())

    return selected_atoms


def get_minimal_core(
    mol: Chem.Mol, atom_ids: Iterable[int], *, remove_hs: bool = True
) -> Chem.Mol:
    """
    Extract the minimal substructure of the molecule containing the passed atoms

    The structure is minimal in the sense that it is still a valid molecule and
    preserve the geometry of the core. In practice:
      - rings (aromatic and aliphatic) are preserved in their entirety
      - aromaticity is preserved, which means also selecting some extracyclic
        atoms when necessary. Currently, we select:

        - atom double-bounded to an aromatic carbon (they alter aromaticity)

    The returned mol object should preserve all molecule properties of the starting
    molecule. Some atoms may be transmuted to hydrogens to satisfy RdKit's aromaticity
    model (e.g. exocyclic atoms bounded to aromatic nitrogens). Those are not deemed part
    of the minimal core, but their conversion is necessary to get a valid structure.

    This functionc an typically be used to extract a contact-scaffold: the minimal
    substructure of the molecule responsible for certain contacts or binding model.
    This scaffold can be used e.g. in clustering molecules, estimating diversity etc...


    Parameters
    ----------
    mol : Chem.Mol
        The molecule to extract a minimal core from. It is NOT re-indexed so that
        atoms have canonical indexing. The passed indexing is preserved. This is
        necessary, as some application requires using non-canonical atom IDs.
    atom_ids : Iterable[int]
        RdKit's internal atom IDs of the atom in ``mol`` to derive a minimal core
        from

    Returns
    -------
    Chem.Mol
        _description_

    Raises
    ------
    ImportError
        _description_
    """
    try:
        import networkx  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "The get_minimal_core() function requires the networkx package. "
        ) from err
    selected_atoms = _get_minimal_core_atomids(mol, list(atom_ids))

    # Mutate the molecule by deleting un-selected atoms
    rwmol = Chem.RWMol(Chem.Mol(mol))
    rwmol.BeginBatchEdit()
    # First, we add hydrogens on the selected atom for each removed
    # neighbors, to preserve valence
    added_hs = set()
    for atom_id in selected_atoms:
        for neighbor in rwmol.GetAtomWithIdx(atom_id).GetNeighbors():
            if neighbor.GetIdx() not in selected_atoms:
                bond = rwmol.GetBondBetweenAtoms(atom_id, neighbor.GetIdx())
                arity = int(bond.GetBondTypeAsDouble())
                for _ in range(arity):
                    hs_id = rwmol.AddAtom(Chem.Atom("H"))
                    rwmol.AddBond(atom_id, hs_id, Chem.BondType.SINGLE)
                    added_hs.add(hs_id)
    # Then delet the non-selected atoms
    for atom in rwmol.GetAtoms():
        atom_id = atom.GetIdx()
        if atom_id not in selected_atoms and atom_id not in added_hs:
            rwmol.RemoveAtom(atom_id)
    rwmol.CommitBatchEdit()

    # Clean-up our selection
    result = rwmol.GetMol()
    Chem.SanitizeMol(result)
    # Remove our hydrogens. The tautomers are stored on the protonated atoms,
    # they explicitely have a hydrogen (it's a property of the atom), so this
    # should not change tautomer
    if remove_hs:
        result = Chem.RemoveHs(result)
    return result

