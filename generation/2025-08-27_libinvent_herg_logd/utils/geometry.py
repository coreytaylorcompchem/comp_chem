from __future__ import annotations

import dataclasses
import functools
from typing import Iterable, Optional, Protocol, Union

import numpy as np
from numpy import typing as nptypes
from rdkit import Chem, Geometry
from rdkit.Chem import rdMolTransforms
from scipy.spatial import transform


class MoleculeTransformProtocol(Protocol):
    """
    Protocol for molecular transformation

    This describe the API (that is, the attributes, properties and methods,
    as well as their arguments and return types) that a molecule transformation
    must respect.
    """

    @property
    def inverse(self) -> MoleculeTransformProtocol:
        """Retrieve the inverse molecular transformation"""
        ...

    def __call__(self, /, mol: Chem.Mol, *, conformer_id: int = 0, copy: bool = True) -> Chem.Mol:
        """
        Apply the transformation to the passed molecule

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to transform
        conformer_id : int, optional
            The RdKit internal ID of the conformer to transform, by default 0
        copy : bool, optional
            Whether to copy the molecule before transformation, by default True

        Returns
        -------
        Chem.Mol
            The transformed molecule. If ``copy=True``, this is a copy, else the original
            molecule is transformed in-place and returned
        """
        ...


def _as_3d_vector(value: nptypes.ArrayLike) -> np.ndarray:
    """Convert a value to a 3D vector"""
    array = np.array(value, dtype=float)
    if array.shape == (3,):
        pass
    elif array.shape in ((3, 1), (1, 3)):
        array = np.squeeze(array)
    elif array.shape == (3, 3) and np.all(array == np.diag(np.diagonal(array))):
        array = np.diagonal(array)
    elif (
        array.shape == (3, 3)
        and np.all(array == np.diag(np.diagonal(array)))
        and np.abs(array[3, 3]) < 1e-12
    ):
        array = np.diagonal(array)[:3]
    else:
        raise ValueError(
            f"Expected 3d vector as, got {array.shape}-shaped "
            f"{type(value).__name__} object {value}"
        )
    return array


@dataclasses.dataclass(frozen=True)
class MatrixTransform(MoleculeTransformProtocol):
    """
    Apply a matrix transformation in homogeneous coordinates to a conformer

    This class implements the MoleculeTransformProtocol, and additionally
    supports restricting the transformation to a subset of atoms.

    Homogeneous coordinates are a way to represent linear and affine 3D
    transformation, that is, it can do translation in addition to rotation
    and axis flips. Homogeneous coordinates also allow a differenciation
    between points and vectors, so that translation have no effect on vectors
    but do affect points. See
    https://en.wikipedia.org/wiki/Homogeneous_coordinates
    for details.

    You can multiply matrix transform together to get a new matrix transform
    as if you multiplied the underlying matrices. The operator for that is the
    matrix multiplication operator ``@``.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix of the transform to apply. It can be either a 3x3 matrix
        of a standard 3D transform, or a 4x4 matrix in homogeneous coordinates.

    Examples
    --------
        >>> yflip = MatrixTransform.mirror(y=True)
        >>> xflip = MatrixTransform.mirror(x=True)
        # Invert y-axis, then x-axis. Using mirror direclty would do
        # the reverse order
        >>> yxflip = yflip @ xflip
        >>> yxflip(mol)
    """

    matrix: np.ndarray
    _matrix: np.ndarray = dataclasses.field(init=False, repr=False, compare=False)

    # Make the array "immutable" by always returning a copy
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        # This will not happen, because dataclass is frozen so @dataclass
        # prevents setting attributes after __init__
        if hasattr(self, "_matrix"):
            raise RuntimeError()
        # by-pass the frozen-ness, like @dataclass does in its generated __init__
        object.__setattr__(self, "_matrix", value)

    def __post_init__(self):
        # accept 3x3 matrices, transform them to homogeneous coords
        if self._matrix.shape == (3, 3):
            temp = np.identity(4)
            temp[:3, :3] = self._matrix
            object.__setattr__(self, "_matrix", temp)
        elif self._matrix.shape != (4, 4):
            raise ValueError(
                f"Expected transformation matrix with shape (4, 4), got {self._matrix.shape}"
            )

    def __matmul__(self, other: MatrixTransform) -> MatrixTransform:
        if isinstance(other, MatrixTransform):
            return MatrixTransform(self._matrix @ other._matrix)
        return NotImplemented

    @classmethod
    def mirror(cls, x: bool = False, y: bool = False, z: bool = False) -> MatrixTransform:
        """
        Make a MatrixTransform object for axis mirroring

        If multiple axes are set to true, the transformations are applied
        in the (x, y, z) order.

        Parameters
        ----------
        x : bool, optional
            Inverse the x-axis, by default False
        y : bool, optional
            Inverse the y-axis, by default False
        z : bool, optional
            Inverse the z-axis, by default False

        Returns
        -------
        MatrixTransform
            A matrix transform object for the transformation
        """
        diag = np.array([x, y, z], dtype=float)
        diag = -2 * diag + 1
        return cls(np.diag(diag))

    @classmethod
    def rotate(
        cls, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None
    ) -> MatrixTransform:
        """
        Make a transformation that applies rotations around the base axes.

        If multiple axes are rotated, the transformations are applied
        in the (x, y, z) order.

        The angles are in radians.

        Parameters
        ----------
        x : Optional[float], optional
            Rotation angle around the x-axis, in radians, by default None
        y : Optional[float], optional
            Rotation angle around the y-axis, in radians, by default None
        z : Optional[float], optional
            Rotation angle around the z-axis, in radians, by default None

        Returns
        -------
        MatrixTransform
            A matrix transform applying the specified rotations
        """

        matrix = np.identity(3, dtype=float)
        if x is not None:
            matrix = matrix @ transform.Rotation.from_euler("x", x).as_matrix()
        if y is not None:
            matrix = matrix @ transform.Rotation.from_euler("y", y).as_matrix()
        if z is not None:
            matrix = matrix @ transform.Rotation.from_euler("z", z).as_matrix()

        return cls(matrix)

    @classmethod
    def rotate_axis(
        cls, axis: nptypes.ArrayLike, radians: Optional[float] = None
    ) -> MatrixTransform:
        """
        Make a transform for a rotation around an axis

        Parameters
        ----------
        axis : np.ndarray
            A 3D vector to rotate around. Can be a (3), (1x3) or (3x1) array,
            a (3x3) diagonal matrix or a (4x4) diagonale matrix with 0 in
            the bottom right corner (i.e. a vector in homogenous coordinates).
        radians : Optional[float], optional
            Optionally the magnitude of the rotation, in radians, by default None.
            If provided, the provided axis is normalized to unit length, and then
            this angle is used. Else, the length of the axis is used as the magnitude
            of the rotation

        Returns
        -------
        MatrixTransform
            A transform for the rotation
        """
        axis = _as_3d_vector(axis)
        if radians is not None:
            axis = axis / np.linalg.norm(axis) * radians
        return MatrixTransform(transform.Rotation.from_rotvec(axis).as_matrix())

    @classmethod
    def translate(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> MatrixTransform:
        """
        Make a MatrixTransform for translation

        Parameters
        ----------
        x : float, optional
            Translation along the x-axis, by default 0.0
        y : float, optional
            Translation along the y-axis, by default 0.0
        z : float, optional
            Translation along the z-axis, by default 0.0

        Returns
        -------
        MatrixTransform
            A matrix transformation for the translation
        """
        # Use homogeneous coordinates to represent a translation
        matrix = np.identity(4, dtype=float)
        for axis, amount in enumerate([x, y, z]):
            matrix[axis, 3] = amount
        return cls(matrix)

    @classmethod
    def translate_vector(cls, vector: nptypes.ArrayLike):
        """
        Make a MatrixTransform object for a translation of a particular vector

        Parameters
        ----------
        vector : nptypes.ArrayLike
            The translationt o apply as a 3D vector. Can be a (3), (1x3) or (3x1) array,
            a (3x3) diagonal matrix or a (4x4) diagonale matrix with 0 in
            the bottom right corner (i.e. a vector in homogenous coordinates).

        Returns
        -------
        _type_
            _description_
        """
        matrix = np.identity(4, dtype=float)
        matrix[:3, 3] = _as_3d_vector(vector)
        return MatrixTransform(matrix)

    @functools.cached_property
    def inverse(self) -> MatrixTransform:
        """
        Return the inverse transform

        The inverse transform is such that applying a transfrom followed by
        its inverse, the conformer geometry is left unchanged (up to rounding
        errors due to floats)

        Returns
        -------
        MatrixTransform
            The inverse transform
        """
        inverse = np.identity(4, dtype=float)
        rot_inv = self._matrix[:3, :3].T
        inverse[:3, :3] = rot_inv
        inverse[:3, 3] = -rot_inv @ self._matrix[:3, 3]
        return type(self)(inverse)

    def __call__(
        self,
        /,
        mol: Chem.Mol,
        *,
        conformer_id: int = 0,
        copy: bool = True,
        atom_ids: Optional[Iterable[int]] = None,
    ) -> Chem.Mol:
        """
        Apply the matrix transformation to the passed molecule

        Parameters
        ----------
        mol : Chem.Mol
            Molecule to apply the transform to
        conformer_id : int, optional
            The ID of the conformer of the molecule to transform, by default 0
        copy : bool, optional
            Whether to return a copy of the molecule instead of modifying in-place, by default True
        atom_ids : Optional[Iterable[int]], optional
            Restrict the transformation to a subset of the atoms, by default None. If None,
            all atoms are transformed.

        Returns
        -------
        Chem.Mol
            The transformed molecule. This is the original molecule transformed in-place if
            copy=False, else a new molecule.
        """
        if copy:
            mol = Chem.Mol(mol)
        conformer = mol.GetConformer(conformer_id)
        # Use the quick RdKit implementation if we can
        if atom_ids is None:
            rdMolTransforms.TransformConformer(conformer, self._matrix)
        else:
            # We used atom filtering: apply our own implementation
            atom_ids = list(atom_ids)
            num_atoms = mol.GetNumAtoms()
            invalid_ids = [idx for idx in atom_ids if not (0 < idx < num_atoms)]
            if invalid_ids:
                raise ValueError(
                    f"Invalid atom IDs, molecule has {num_atoms} atoms: {invalid_ids}"
                )
            # This array has shape 4 because we use homogeneous coordinates
            # See https://en.wikipedia.org/wiki/Homogeneous_coordinates
            # last component is 1 for points
            atom_positions = np.ones((len(atom_ids), 4), dtype=float)  # shape: (n_atom, 4)
            # Inject the 3D coordinates into the homogeneous array
            atom_positions[:, :3] = conformer.GetPositions()[atom_ids]
            # Apply our transformation, and normalize
            # We need to add a dummy dimension at the end of the vector stack
            # so that numpy broadcasts the matrix transformation to each vector,
            # Without that, numpy would do 2D matrix multiplication
            atom_positions = self._matrix @ atom_positions[:, :, None]
            # Normalize the homogeneous dimension
            atom_positions = atom_positions / atom_positions[:, 3, None]
            # Apply the position to the conformer
            for atom_id, pos in zip(atom_ids, atom_positions[:, :3]):
                conformer.SetAtomPosition(atom_id, Geometry.Point3D(*map(float, pos)))
        return mol


mirror = MatrixTransform.mirror
rotate = MatrixTransform.rotate
rotate_axis = MatrixTransform.rotate_axis
translate = MatrixTransform.translate
translate_vector = MatrixTransform.translate_vector


@dataclasses.dataclass(frozen=True)
class BondRotationTransform(MoleculeTransformProtocol):
    """
    Transformation to rotate a rotatable bond

    The part of the molecule that is rotated is the part toward the ``end`` atom.
    Flipping the start and end atom only change which part of the molecule
    is rotated, but the sign of the rotation is preserved.

    Parameters
    ----------
    start_atom_idx : int
        RdKit index of start atom of the bond to rotate around
    end_atom_idx : int
        RdKit index of the end atom of the bond to rotate around
    radians : Optional[float], optional
        The rotation angle, in radians.
        Exactly one of ``radians`` or ``degrees`` must be provided
    degrees : Optional[float], optional
        The rotation angle, in degrees.
        Exactly one of ``radians`` or ``degrees`` must be provided
    map_index_from_substruct : Union[None, str, Chem.Mol]
        Substructure to use to specify the indexes. This allows rotating a bond on many
        molecules by specifying the indexes in a shared substructure and using the same
        indexes, rather than having to change the index every time.
    """

    start_atom_idx: int
    end_atom_idx: int
    radians: float
    map_index_from_substruct: Union[None, Chem.Mol] = None

    def __init__(
        self,
        start_atom_idx: int,
        end_atom_idx: int,
        *,
        radians: Optional[float] = None,
        degrees: Optional[float] = None,
        map_index_from_substruct: Union[None, str, Chem.Mol] = None,
    ) -> None:
        setattr = object.__setattr__
        setattr(self, "start_atom_idx", start_atom_idx)
        setattr(self, "end_atom_idx", end_atom_idx)

        # Parse the angle
        if radians is not None and degrees is not None:
            raise TypeError("Arguments 'radians' and 'degrees' are incompatible")
        elif radians is not None:
            pass
        elif degrees is not None:
            radians = degrees / 180 * np.pi
        else:
            raise TypeError(
                "__init__() missing 1 required keyword-onky argument: 'radians' or 'degrees'"
            )
        # Preserve the sign of the rotation if indexes are swapped
        radians = radians if end_atom_idx > start_atom_idx else -radians
        setattr(self, "radians", radians)

        # Parse substructure
        substruct = None
        if map_index_from_substruct is not None:
            if isinstance(map_index_from_substruct, Chem.Mol):
                substruct = map_index_from_substruct
            elif isinstance(map_index_from_substruct, str):
                # Simple way to differenciate JSON from SMILES
                if "{" in map_index_from_substruct:
                    mols = Chem.JSONToMols(map_index_from_substruct)
                    if len(mols) != 1:
                        raise ValueError(
                            "argument 'map_index_from_substruct' doesn't contain exactly one molecule"
                        )
                    substruct = mols[0]
                else:
                    substruct = Chem.MolFromSmiles(map_index_from_substruct)
            else:
                raise TypeError(
                    "Expected 'map_index_from_substruct' with type None, str or rdkit.Chem.Mol, "
                    f"got {type(map_index_from_substruct).__name__}"
                )
            setattr(self, "map_index_from_substruct", substruct)

        # If substruct is provided, check the atom IDs
        if substruct is not None:
            num_atoms = substruct.GetNumAtoms()
            for idx in (start_atom_idx, end_atom_idx):
                if not (0 <= idx < num_atoms):
                    raise ValueError(
                        f"Invalid atom index {idx} for {map_index_from_substruct}: "
                        f"molecule has {num_atoms} atoms."
                    )

    @functools.cached_property
    def inverse(self) -> BondRotationTransform:
        """Produce the inverse transformation, that simply rotate the same bond the other way"""
        return BondRotationTransform(
            start_atom_idx=self.start_atom_idx,
            end_atom_idx=self.end_atom_idx,
            radians=-self.radians,
            map_index_from_substruct=self.map_index_from_substruct,
        )

    def __call__(self, /, mol: Chem.Mol, *, conformer_id: int = 0, copy: bool = True) -> Chem.Mol:
        """
        Apply the bond rotation to the molecule

        Parameters
        ----------
        mol : Chem.Mol
            Molecule to rotate the bond of
        conformer_id : int, optional
            ID of the conformer to modify, by default 0
        copy : bool, optional
            Whether to modify the molecule in-place or a copy, by default True (a copy).

        Returns
        -------
        Chem.Mol
            The modified molecule
        """
        if copy:
            mol = Chem.Mol(mol)
        start, end = self.start_atom_idx, self.end_atom_idx
        # Map the atom ids if applicable
        if self.map_index_from_substruct is not None:
            matches = mol.GetSubstructMatches(self.map_index_from_substruct)
            if len(matches) < 1:
                raise ValueError(
                    f"{Chem.MolToSmiles(mol)} doesn't match "
                    f"{Chem.MolToSmiles(self.map_index_from_substruct)}"
                )
            elif len(matches) == 1:
                atom_map = matches
                start, end = atom_map[start], atom_map[end]
            else:
                raise ValueError(
                    f"{Chem.MolToSmiles(mol)} has multiple matches for "
                    f"{Chem.MolToSmiles(self.map_index_from_substruct)}"
                )
        # Check atom indexes
        num_atoms = mol.GetNumAtoms()
        for idx in (start, end):
            if not (0 <= idx < num_atoms):
                raise ValueError(f"Invalid atom ID, molecule has {num_atoms} atoms: {idx}")

        # Extract the basic data
        conformer = mol.GetConformer(conformer_id)
        start_point = np.array(conformer.GetAtomPosition(start))
        end_point = np.array(conformer.GetAtomPosition(end))

        # Compute the rotation we want to apply
        rotation = MatrixTransform.rotate_axis(end_point - start_point, self.radians)

        # To rotate the bond, we need to have the atom that doesn't move, 'end', at the origin
        # We compose our rotation with the two necessary translation
        # This is because points are not vectors, so a rotation is not the same everywhere
        # in the plan. Said differently, a vector-space is a linear space, while a point-space
        # is an affine space
        translation = MatrixTransform.translate_vector(-end_point)
        transform = translation.inverse @ rotation @ translation

        # Now, find the atoms that need rotating
        atom_ids = {neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(end).GetNeighbors()} | {
            end
        }
        if start not in atom_ids:
            raise ValueError(f"Atom #{start} and #{end} are not connected")
        # Now keep adding neighbors of new atoms until we have the whole part that rotates
        atom_ids -= {start}
        new_atoms = atom_ids - {end}
        while new_atoms:
            more_atoms = {
                neighbor.GetIdx()
                for atom_id in new_atoms
                for neighbor in mol.GetAtomWithIdx(atom_id).GetNeighbors()
            }
            more_atoms -= new_atoms
            more_atoms -= atom_ids
            if start in more_atoms:
                raise ValueError(
                    f"Cycle detected from #{end} to #{start}, is (#{start}, #{end}) really "
                    "a rotatable bond ?"
                )
            atom_ids |= more_atoms
            new_atoms = more_atoms

        # Build & apply our rotation
        return transform(mol, conformer_id=conformer_id, atom_ids=atom_ids)
