"""
preprocess_custom.py
Reconstructed preprocessing script for ML-PLA.
Works on raw PDBBind v2016 data structured as:
    v2016/
        1a1e/
            1a1e_protein.pdb
            1a1e_pocket.pdb
            1a1e_ligand.mol2
            1a1e_ligand.sdf
        ...
    index/
        INDEX_general_PL_data.2016   <- affinity labels

Outputs (written to ./data/):
    mol_dicts/    <- pickled (mol1=ligand, mol2=pocket) pairs per complex
    residue_dic/  <- pickled residue DGL graphs per protein
    processed/    <- final graph list .bin files for training
    all_protein_files/ <- protein pdb copies
    all_protein.csv    <- protein ID list
"""

import os
import sys
import re
import pickle
import math
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from itertools import repeat
from functools import partial

from rdkit import Chem
from rdkit.Chem import AllChem

import dgl


# ─────────────────────────────────────────────
# STEP 0: Configuration — edit these paths
# ─────────────────────────────────────────────

PDBBIND_ROOT = '/kaggle/input/datasets/maithreyadonthi/pdbbind-v2016/v2016'
INDEX_FILE   = '/kaggle/input/datasets/maithreyadonthi/pdbbind-v2016/v2016/index/INDEX_general_PL_data.2016'
OUTPUT_DIR   = '/kaggle/working/ML-PLA/data'
NUM_PROCESS  = 4          # number of CPU cores for multiprocessing
DIS_THRESHOLD = 5.0       # Angstrom threshold for protein-ligand interaction edges
KNN_K        = 5          # K for KNN residue graph edges
RES_THRESHOLD = 10.0      # Angstrom threshold for residue contact edges


# ─────────────────────────────────────────────
# STEP 1: Parse affinity labels from INDEX file
# ─────────────────────────────────────────────

def parse_index(index_file):
    """
    Parse the PDBBind INDEX file.
    Returns dict: {pdb_id: pKa_value}
    Lines look like:
    1a1e  2007  1.80  Ki=1.5nM   // ...
    """
    data = {}
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split()
            pdb_id = parts[0].lower()
            try:
                pka = float(parts[3])   # column 4 = -log(affinity)
                data[pdb_id] = pka
            except (IndexError, ValueError):
                continue
    print(f"Parsed {len(data)} complexes from index.")
    return data


# ─────────────────────────────────────────────
# STEP 2: Read ligand + pocket → pickle (mol1, mol2)
# ─────────────────────────────────────────────

def read_mol(pdb_id, pdbbind_root):
    """
    Read ligand (mol2 or sdf) and pocket (pdb) for a given complex.
    Returns (ligand_mol, pocket_mol) as RDKit Mol objects with 3D coordinates.
    """
    folder = os.path.join(pdbbind_root, pdb_id)

    # Ligand: prefer .mol2, fallback to .sdf
    mol2_path = os.path.join(folder, f'{pdb_id}_ligand.mol2')
    sdf_path  = os.path.join(folder, f'{pdb_id}_ligand.sdf')

    ligand = None
    if os.path.exists(mol2_path):
        ligand = Chem.MolFromMol2File(mol2_path, removeHs=True)
    if ligand is None and os.path.exists(sdf_path):
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
        for m in suppl:
            if m is not None:
                ligand = m
                break

    # Pocket
    pocket_path = os.path.join(folder, f'{pdb_id}_pocket.pdb')
    pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True, sanitize=False)
    if pocket is not None:
        try:
            Chem.SanitizeMol(pocket)
        except Exception:
            pass

    return ligand, pocket


def save_mol_pickle(pdb_id, pdbbind_root, out_dir):
    """Convert one complex to (mol1, mol2) pickle."""
    try:
        mol1, mol2 = read_mol(pdb_id, pdbbind_root)
        if mol1 is None or mol2 is None:
            return False
        if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
            return False
        out_path = os.path.join(out_dir, pdb_id)
        with open(out_path, 'wb') as f:
            pickle.dump((mol1, mol2), f)
        return True
    except Exception as e:
        print(f"  [SKIP] {pdb_id}: {e}")
        return False


# ─────────────────────────────────────────────
# STEP 3: Build residue-level heterogeneous graph
# (mirrors pdb_to_cm from graph_constructor.py)
# ─────────────────────────────────────────────

AMINO_ACIDS = ['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU',
               'MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']

def dist(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def read_atoms(file_name):
    """Extract Cα atom coordinates and residue types from a PDB file."""
    atoms, ajs = [], []
    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_type = line[12:16].strip()
                if atom_type == "CA":
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    ajs_id = line[17:20].strip()
                    atoms.append((x, y, z))
                    ajs.append(ajs_id)
    return atoms, ajs

def match_feature(ajs, all_for_assign):
    """Map residue names to feature vectors."""
    x_p = np.zeros((len(ajs), 7))
    for j, aa in enumerate(ajs):
        if aa in AMINO_ACIDS:
            x_p[j] = all_for_assign[AMINO_ACIDS.index(aa), :]
    return x_p

def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms) - 2):
        for j in range(i + 2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j)); contacts.append((j, i))
    return contacts

def knn(atoms, k=5):
    n = len(atoms)
    if n == 0:
        return []
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = dist(atoms[i], atoms[j])
    index = np.argsort(dist_matrix, axis=-1)
    contacts = []
    for i in range(n):
        count = 0
        for j in range(n):
            nb = index[i, j]
            if nb != i and nb != i - 1 and nb != i + 1:
                contacts.append((i, nb))
                count += 1
            if count == k:
                break
    return contacts

def build_residue_graph(pdb_id, pdbbind_root, out_dir, all_for_assign,
                        threshold=10.0, k=5):
    """Build and save residue-level heterogeneous DGL graph for one protein."""
    pdb_path = os.path.join(pdbbind_root, pdb_id, f'{pdb_id}_protein.pdb')
    if not os.path.exists(pdb_path):
        return False
    try:
        atoms, ajs = read_atoms(pdb_path)
        if len(atoms) < 3:
            return False
        x_p = match_feature(ajs, all_for_assign)
        r_contacts = compute_contacts(atoms, threshold)
        k_contacts = knn(atoms, k)
        seq_edges = [(j, j+1) for j in range(len(atoms)-1)] + \
                    [(j+1, j) for j in range(len(atoms)-1)]

        # Need at least 1 edge per relation type for DGL heterograph
        if not r_contacts:
            r_contacts = [(0, 0)]
        if not k_contacts:
            k_contacts = [(0, 0)]

        g = dgl.heterograph({
            ('amino_acid', 'SEQ',     'amino_acid'): seq_edges,
            ('amino_acid', 'STR_KNN', 'amino_acid'): k_contacts,
            ('amino_acid', 'STR_DIS', 'amino_acid'): r_contacts,
        })
        g.ndata['x'] = torch.FloatTensor(x_p)

        with open(os.path.join(out_dir, pdb_id), 'wb') as f:
            pickle.dump(g, f)
        return True
    except Exception as e:
        print(f"  [RES SKIP] {pdb_id}: {e}")
        return False


# ─────────────────────────────────────────────
# STEP 4: Build atom-level graphs (g1, g2, g3)
# (mirrors graphs_from_mol_mul from graph_constructor.py)
# ─────────────────────────────────────────────

from dgllife.utils import (BaseAtomFeaturizer, atom_type_one_hot,
    atom_degree_one_hot, atom_total_num_H_one_hot, atom_is_aromatic,
    ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot,
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons,
    bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot,
    BaseBondFeaturizer)
from scipy.spatial import distance_matrix as sp_dist_matrix

def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False, atom.HasProp('_ChiralityPossible')]

class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self):
        super().__init__(featurizer_funcs={'h': ConcatFeaturizer([
            partial(atom_type_one_hot,
                    allowable_set=['C','N','O','S','F','P','Cl','Br','I',
                                   'B','Si','Fe','Zn','Cu','Mn','Mo'],
                    encode_unknown=True),
            partial(atom_degree_one_hot, allowable_set=list(range(6))),
            atom_formal_charge, atom_num_radical_electrons,
            partial(atom_hybridization_one_hot, encode_unknown=True),
            atom_is_aromatic, atom_total_num_H_one_hot, chirality])})

class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self):
        super().__init__(featurizer_funcs={'e': ConcatFeaturizer([
            bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
            partial(bond_stereo_one_hot,
                    allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                   Chem.rdchem.BondStereo.STEREOANY,
                                   Chem.rdchem.BondStereo.STEREOZ,
                                   Chem.rdchem.BondStereo.STEREOE],
                    encode_unknown=True)])})

AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer  = MyBondFeaturizer()

def mol_to_graph(mol):
    """Build a DGL graph from an RDKit mol."""
    g = dgl.DGLGraph()
    n = mol.GetNumAtoms()
    g.add_nodes(n)
    srcs, dsts = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        srcs += [u, v]; dsts += [v, u]
    if srcs:
        g.add_edges(srcs, dsts)
    g.ndata['h'] = AtomFeaturizer(mol)['h']
    pos = torch.tensor(mol.GetConformers()[0].GetPositions(), dtype=torch.float)
    if g.number_of_edges() > 0:
        dm = sp_dist_matrix(pos.numpy(), pos.numpy())
        d = torch.tensor(dm[srcs, dsts], dtype=torch.float).view(-1,1)
        efeats = BondFeaturizer(mol)['e']
        g.edata['e'] = torch.cat([torch.cat([efeats[::2], efeats[::2]]), d * 0.1], dim=-1)
    g.ndata['pos'] = pos
    return g

def build_complex_graph(pdb_id, mol_dir, residue_dir, graph_dir,
                        dis_threshold=5.0):
    """Build g1, g2, g3, g4 and save as pickle."""
    try:
        with open(os.path.join(mol_dir, pdb_id), 'rb') as f:
            mol1, mol2 = pickle.load(f)
        residue_path = os.path.join(residue_dir, pdb_id)
        if not os.path.exists(residue_path):
            print(f"  [NO RESIDUE] {pdb_id}"); return False

        with open(residue_path, 'rb') as f:
            g4 = pickle.load(f)

        g1 = mol_to_graph(mol1)
        g2 = mol_to_graph(mol2)

        pos1 = mol1.GetConformers()[0].GetPositions()
        pos2 = mol2.GetConformers()[0].GetPositions()
        dm   = sp_dist_matrix(pos1, pos2)
        idx  = np.where(dm < dis_threshold)
        n1, n2 = mol1.GetNumAtoms(), mol2.GetNumAtoms()
        srcs3 = np.concatenate([idx[0], idx[1] + n1])
        dsts3 = np.concatenate([idx[1] + n1, idx[0]])
        g3 = dgl.DGLGraph(); g3.add_nodes(n1 + n2)
        if len(srcs3) > 0:
            g3.add_edges(srcs3, dsts3)
            inter_d = np.concatenate([dm[idx[0], idx[1]], dm[idx[0], idx[1]]])
            g3.edata['e'] = torch.tensor(inter_d, dtype=torch.float).view(-1, 1) * 0.1

        # Remove pos from g1/g2 before saving (as original code does)
        g1.ndata.pop('pos'); g2.ndata.pop('pos')

        out = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4, 'key': pdb_id}
        with open(os.path.join(graph_dir, pdb_id), 'wb') as f:
            pickle.dump(out, f)
        return True
    except Exception as e:
        print(f"  [GRAPH SKIP] {pdb_id}: {e}")
        return False


# ─────────────────────────────────────────────
# STEP 5: Collect graphs into .bin list files
# ─────────────────────────────────────────────

def collect_graphs(graph_dir, label_dict, out_dir):
    g1s, g2s, g3s, g4s, labels, keys = [], [], [], [], [], []
    for key in os.listdir(graph_dir):
        path = os.path.join(graph_dir, key)
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            if key not in label_dict:
                continue
            g1s.append(d['g1']); g2s.append(d['g2'])
            g3s.append(d['g3']); g4s.append(d['g4'])
            labels.append(label_dict[key]); keys.append(key)
        except Exception as e:
            print(f"  [COLLECT SKIP] {key}: {e}")

    os.makedirs(out_dir, exist_ok=True)
    for name, obj in [('g1', g1s), ('g2', g2s), ('g3', g3s),
                      ('residue', g4s), ('labels', labels), ('keys', keys)]:
        with open(os.path.join(out_dir, f'{name}.bin'), 'wb') as f:
            pickle.dump(obj, f)
    print(f"Saved {len(keys)} complexes to {out_dir}")
    return keys


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Read affinity labels
    print("=" * 50)
    print("STEP 1: Parsing affinity labels...")
    label_dict = parse_index(INDEX_FILE)

    # Only keep complexes that exist in the data folder
    all_ids = [d for d in os.listdir(PDBBIND_ROOT)
               if os.path.isdir(os.path.join(PDBBIND_ROOT, d)) and d in label_dict]
    print(f"Found {len(all_ids)} complexes with labels")

    # Create output directories
    mol_dir     = os.path.join(OUTPUT_DIR, 'mol_dicts')
    residue_dir = os.path.join(OUTPUT_DIR, 'residue_dic')
    graph_dir   = os.path.join(OUTPUT_DIR, 'graph_dic')
    proc_dir    = os.path.join(OUTPUT_DIR, 'processed')
    prot_dir    = os.path.join(OUTPUT_DIR, 'all_protein_files')
    for d in [mol_dir, residue_dir, graph_dir, proc_dir, prot_dir]:
        os.makedirs(d, exist_ok=True)

    # Load amino acid feature table (from repo ./data/all_assign.txt)
    assign_path = os.path.join(OUTPUT_DIR, 'all_assign.txt')
    if not os.path.exists(assign_path):
        print("WARNING: all_assign.txt not found — downloading from repo...")
        os.system(f'wget -q https://raw.githubusercontent.com/Biowust/ML-PLA/main/data/all_assign.txt -O {assign_path}')
    all_for_assign = np.loadtxt(assign_path)

    # STEP 2: Ligand + pocket → pickle
    print("=" * 50)
    print("STEP 2: Pickling (ligand, pocket) mol pairs...")
    to_pickle = [pid for pid in all_ids
                 if not os.path.exists(os.path.join(mol_dir, pid))]
    print(f"  {len(to_pickle)} to process (skipping existing)")
    for i, pid in enumerate(to_pickle):
        ok = save_mol_pickle(pid, PDBBIND_ROOT, mol_dir)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(to_pickle)} done")

    valid_ids = [pid for pid in all_ids if os.path.exists(os.path.join(mol_dir, pid))]
    print(f"  {len(valid_ids)} complexes successfully pickled")

    # STEP 3: Residue graphs
    print("=" * 50)
    print("STEP 3: Building residue graphs...")
    to_residue = [pid for pid in valid_ids
                  if not os.path.exists(os.path.join(residue_dir, pid))]
    print(f"  {len(to_residue)} to process")
    for i, pid in enumerate(to_residue):
        build_residue_graph(pid, PDBBIND_ROOT, residue_dir,
                            all_for_assign, threshold=RES_THRESHOLD, k=KNN_K)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(to_residue)} done")

    valid_res = [pid for pid in valid_ids if os.path.exists(os.path.join(residue_dir, pid))]
    print(f"  {len(valid_res)} residue graphs built")

    # Also copy protein pdb files and make CSV (needed by pretrain_vae)
    print("Copying protein pdb files...")
    for pid in valid_res:
        src = os.path.join(PDBBIND_ROOT, pid, f'{pid}_protein.pdb')
        dst = os.path.join(prot_dir, f'{pid}_protein.pdb')
        if not os.path.exists(dst) and os.path.exists(src):
            import shutil; shutil.copy(src, dst)
    pd.DataFrame({'id': valid_res}).to_csv(
        os.path.join(OUTPUT_DIR, 'all_protein.csv'), index=False)
    print(f"Saved all_protein.csv with {len(valid_res)} entries")

    # STEP 4: Full complex graphs
    print("=" * 50)
    print("STEP 4: Building complex atom graphs...")
    to_graph = [pid for pid in valid_res
                if not os.path.exists(os.path.join(graph_dir, pid))]
    print(f"  {len(to_graph)} to process")
    for i, pid in enumerate(to_graph):
        build_complex_graph(pid, mol_dir, residue_dir, graph_dir, DIS_THRESHOLD)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(to_graph)} done")

    # STEP 5: Collect into .bin files
    print("=" * 50)
    print("STEP 5: Collecting graphs into .bin files...")
    final_keys = collect_graphs(graph_dir, label_dict, proc_dir)

    print("=" * 50)
    print(f"✅ PREPROCESSING COMPLETE — {len(final_keys)} complexes ready")
    print(f"   Output at: {proc_dir}")
    print(f"   Files: g1.bin, g2.bin, g3.bin, residue.bin, labels.bin, keys.bin")
