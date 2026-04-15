"""
qm_features.py
--------------
Loads quantum-mechanical (QM) descriptors from the MISATO QM HDF5 file
and exposes them for use in the ML-PLA graph-construction pipeline.

Per-atom features selected (25 dims, after dropping x/y/z):
  hybridisation, group,
  gfn2_charge, gfn2_charge_(wet_octanol), gfn2_charge_(water),
  AM1_charge, AM1_CM1_charge, AM1_CM2_charge, AM1_CM3_charge, PM6_charge,
  gfn2_polarisation, gfn2_polarisation_(wet_octanol), gfn2_polarisation_(water),
  gfn2_charge_electrophilicity, gfn2_charge_electrophilicity_softness,
  gfn2_charge_nucleophilicity_softness, gfn2_charge_nucleophilicity,
  gfn2_charge_radical, gfn2_charge_radical_softness,
  gfn2_orbital_electrophilicity, gfn2_orbital_electrophilicity_softness,
  gfn2_orbital_nucleophilicity_softness, gfn2_orbital_nucleophilicity,
  gfn2_orbital_radical, gfn2_orbital_radical_softness

Per-molecule features (7 dims):
  Electron_Affinity, Electronegativity, Hardness,
  Ionization_Potential, Koopman, molecular_weight, total_charge
"""

import h5py
import numpy as np
import torch

# ── Dimension constants (importable by other modules) ────────────────────────
QM_ATOM_FEAT_DIM = 25   # per-atom QM descriptors (no xyz)
QM_MOL_FEAT_DIM  = 7    # per-molecule scalar QM properties
MOL_PROP_KEYS = [
    'Electron_Affinity', 'Electronegativity', 'Hardness',
    'Ionization_Potential', 'Koopman', 'molecular_weight', 'total_charge',
]


class QMFeatureLoader:
    """
    Lazy-loaded QM feature cache.

    Usage
    -----
    loader = QMFeatureLoader('path/to/QM.hdf5')

    # Per-atom features aligned to RDKit atom ordering (may need re-ordering)
    atom_feats = loader.get_atom_feats('1ABC')   # np.ndarray (N_atoms, 25)

    # Per-molecule scalar descriptors
    mol_feats  = loader.get_mol_feats('1ABC')    # np.ndarray (7,)
    """

    def __init__(self, hdf5_path: str):
        self._path = hdf5_path
        self._file = None          # opened lazily
        self._mol_cache: dict = {}
        self._atom_cache: dict = {}

    # ── internal helpers ─────────────────────────────────────────────────────

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self._path, 'r')

    def _key_variants(self, pdb_id: str):
        """Return candidate keys (original case + upper + lower)."""
        return [pdb_id, pdb_id.upper(), pdb_id.lower()]

    def _find_key(self, pdb_id: str):
        self._open()
        for k in self._key_variants(pdb_id):
            if k in self._file:
                return k
        return None

    # ── public API ───────────────────────────────────────────────────────────

    def has_entry(self, pdb_id: str) -> bool:
        return self._find_key(pdb_id) is not None

    def get_atom_feats(self, pdb_id: str) -> np.ndarray:
        """
        Returns atom-level QM features (N_atoms, 25) as float32.
        Columns 0-24 correspond to the non-xyz atom properties in order.
        Returns None if the PDB ID is not present.
        """
        if pdb_id in self._atom_cache:
            return self._atom_cache[pdb_id]
        k = self._find_key(pdb_id)
        if k is None:
            return None
        vals = self._file[k]['atom_properties']['atom_properties_values'][:]  # (N, 28)
        feats = vals[:, 3:].astype(np.float32)  # drop x, y, z → (N, 25)
        # replace NaN / Inf with 0, then clip to stable range
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = np.clip(feats, -50.0, 50.0)
        self._atom_cache[pdb_id] = feats
        return feats

    def get_mol_feats(self, pdb_id: str) -> np.ndarray:
        """
        Returns molecule-level QM scalar features (7,) as float32.
        Order: Electron_Affinity, Electronegativity, Hardness,
               Ionization_Potential, Koopman, molecular_weight, total_charge.
        Returns zero vector if the PDB ID is not present.
        """
        if pdb_id in self._mol_cache:
            return self._mol_cache[pdb_id]
        k = self._find_key(pdb_id)
        if k is None:
            feats = np.zeros(QM_MOL_FEAT_DIM, dtype=np.float32)
        else:
            mp = self._file[k]['mol_properties']
            feats = np.array(
                [float(mp[prop][()]) for prop in MOL_PROP_KEYS],
                dtype=np.float32,
            )
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            feats[5] = feats[5] / 100.0  # scale molecular_weight (~300-1500 → 3-15)
            feats = np.clip(feats, -50.0, 50.0)
        self._mol_cache[pdb_id] = feats
        return feats

    def get_mol_feats_tensor(self, pdb_id: str) -> torch.Tensor:
        return torch.tensor(self.get_mol_feats(pdb_id), dtype=torch.float)

    def get_atom_feats_tensor(self, pdb_id: str):
        arr = self.get_atom_feats(pdb_id)
        if arr is None:
            return None
        return torch.tensor(arr, dtype=torch.float)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


# ── Module-level singleton (optional convenience) ────────────────────────────
_global_loader: QMFeatureLoader = None

def init_global_loader(hdf5_path: str):
    global _global_loader
    _global_loader = QMFeatureLoader(hdf5_path)

def get_global_loader() -> QMFeatureLoader:
    if _global_loader is None:
        raise RuntimeError(
            "Call qm_features.init_global_loader(path) before using get_global_loader()."
        )
    return _global_loader
