# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ML-PLA + MISATO QM Features — Full Retraining on Kaggle (GPU T4/P100)     ║
# ║  Run each cell in order. Runtime: ~4–8 hours on T4 for 1000 epochs.        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# BEFORE RUNNING:
# 1. Upload your datasets to Kaggle:
#    - The ML-PLA-Binding-Affinity folder (zipped) as a "Code" dataset
#    - QM (1).hdf5 as a separate dataset (it's 327 MB)
#    - The CASF-2016 + PDBbind data (binding_affinity folder) as a dataset
#
# 2. In your Kaggle notebook settings:
#    - Accelerator: GPU (T4 x2 recommended, or P100)
#    - Internet: ON (for pip installs)
#    - Add your three datasets in the "Data" panel

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Check GPU and paths
# ─────────────────────────────────────────────────────────────────────────────
import os, subprocess

# Print available GPU
subprocess.run(["nvidia-smi"], check=False)

# List your input datasets — adjust these to match your Kaggle dataset slugs
print("\nInput datasets available:")
for d in os.listdir("/kaggle/input"):
    print(f"  /kaggle/input/{d}/")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
# Kaggle already has: torch, numpy, pandas, scipy, sklearn
# We need: dgl, dgllife, rdkit, h5py, prefetch_generator, numba

import subprocess, sys

def pip(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

# DGL with CUDA — match your Kaggle CUDA version (usually 11.8 or 12.1)
# Check with: !nvcc --version  or  torch.version.cuda
import torch
cuda_ver = torch.version.cuda
print(f"PyTorch CUDA version: {cuda_ver}")

if cuda_ver and cuda_ver.startswith("11"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dgl", "-f",
                    "https://data.dgl.ai/wheels/repo.html"], check=True)
else:
    # CUDA 12.x
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dgl", "-f",
                    "https://data.dgl.ai/wheels/cu121/repo.html"], check=True)

pip("dgllife")
pip("rdkit-pypi")
pip("h5py")
pip("prefetch_generator")
pip("numba")
pip("PyYAML")

print("✅ All packages installed.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Set up paths
# ─────────────────────────────────────────────────────────────────────────────

# ── Adjust these to match YOUR Kaggle dataset folder names ──────────────────
MLPLA_INPUT     = "/kaggle/input/ml-pla-binding-affinity"   # your code dataset
QM_HDF5_PATH    = "/kaggle/input/misato-qm/QM (1).hdf5"    # QM HDF5 dataset
BINDING_DATA    = "/kaggle/input/pdbbind-casf2016"          # PDBbind data

# Optional: Resume training from a previous checkpoint (e.g. after a 12h timeout)
# Because you bundled the checkpoint with your code, it will be mapped here:
RESUME_CKPT_PATH = f"{MLPLA_INPUT}/ckpt_ep340.pt"

# Working directory — we'll copy the source here
WORK_DIR        = "/kaggle/working/ML-PLA-Binding-Affinity"
SRC_DIR         = os.path.join(WORK_DIR, "src")
DATA_DIR        = os.path.join(SRC_DIR, "data")
OUTPUT_DIR      = "/kaggle/working/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"MLPLA source  : {MLPLA_INPUT}")
print(f"QM HDF5       : {QM_HDF5_PATH}")
print(f"Binding data  : {BINDING_DATA}")
print(f"Work dir      : {WORK_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Copy source code to working directory
# ─────────────────────────────────────────────────────────────────────────────
import shutil

# Copy the entire project to /kaggle/working so we can write to it
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
shutil.copytree(MLPLA_INPUT, WORK_DIR)
print(f"✅ Copied source to {WORK_DIR}")

# Verify key files exist
for f in [
    os.path.join(SRC_DIR, "trainer", "train.py"),
    os.path.join(SRC_DIR, "dataset", "graph_constructor.py"),
    os.path.join(SRC_DIR, "dataset", "qm_features.py"),
    os.path.join(SRC_DIR, "models", "model.py"),
    os.path.join(SRC_DIR, "configs", "config.yaml"),
]:
    status = "✅" if os.path.exists(f) else "❌ MISSING"
    print(f"  {status}  {f}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Patch config.yaml with Kaggle-correct paths
# ─────────────────────────────────────────────────────────────────────────────
import yaml

CONFIG_PATH = os.path.join(SRC_DIR, "configs", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Update QM path and other Kaggle-specific settings
config["qm_hdf5_path"]    = QM_HDF5_PATH
config["use_qm_mol_feats"] = True
config["num_workers"]     = 2          # Kaggle allows up to 4, keep low for stability
config["batch_size"]      = 64         # reduce if OOM on T4 (16GB VRAM)
config["save_dir"]        = OUTPUT_DIR + "/model_save"
config["epochs"]          = 1000       # uses early stopping (patience=100)

with open(CONFIG_PATH, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ config.yaml updated:")
for k in ["qm_hdf5_path", "use_qm_mol_feats", "batch_size", "num_workers", "epochs", "save_dir"]:
    print(f"   {k}: {config[k]}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Set up data directory symlinks / copies
# ─────────────────────────────────────────────────────────────────────────────
# The train.py script expects data at:
#   ./data/labels_gign12904.csv
#   ./data/labels_casf2016.csv
#   ./data/binding_affinity/train/complex/<PDB_ID>
#   ./data/binding_affinity/validation/complex/<PDB_ID>
#   ./data/binding_affinity/test2016/complex/<PDB_ID>
#   ./data/binding_affinity/processed_data/residue_dic_path/<PDB_ID>
#   ./data/all_protein_files/<PDB_ID>_protein.pdb
#   ./data/all_protein.csv
#   ./data/all_assign.txt

# If your binding_affinity data is at BINDING_DATA, link it:
data_link = DATA_DIR
if not os.path.exists(data_link):
    # Try symlinking first (saves disk space)
    try:
        os.symlink(BINDING_DATA, data_link)
        print(f"✅ Symlinked {BINDING_DATA} → {data_link}")
    except Exception as e:
        print(f"Symlink failed ({e}), copying instead...")
        shutil.copytree(BINDING_DATA, data_link)
        print(f"✅ Copied binding data to {data_link}")
else:
    print(f"ℹ️  Data dir already exists: {data_link}")

# Verify key CSV files
for f in ["labels_gign12904.csv", "labels_casf2016.csv", "all_protein.csv"]:
    path = os.path.join(DATA_DIR, f)
    status = "✅" if os.path.exists(path) else "❌ MISSING"
    print(f"  {status}  {path}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Verify QM loader works in this environment
# ─────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)

from dataset.qm_features import QMFeatureLoader, QM_ATOM_FEAT_DIM, QM_MOL_FEAT_DIM
import numpy as np

loader = QMFeatureLoader(QM_HDF5_PATH)

# Test with a known CASF-2016 entry
test_pdb = "10GS"
af = loader.get_atom_feats(test_pdb)
mf = loader.get_mol_feats(test_pdb)

assert af is not None, f"atom feats for {test_pdb} are None!"
assert af.shape[1] == QM_ATOM_FEAT_DIM, f"Expected 25 atom feats, got {af.shape[1]}"
assert mf.shape[0] == QM_MOL_FEAT_DIM,  f"Expected 7 mol feats, got {mf.shape[0]}"

print(f"✅ QM loader working")
print(f"   Atom feats shape for {test_pdb}: {af.shape}")
print(f"   Mol feats         : {mf}")
print(f"   QM_ATOM_FEAT_DIM  = {QM_ATOM_FEAT_DIM}")
print(f"   QM_MOL_FEAT_DIM   = {QM_MOL_FEAT_DIM}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Quick model sanity check (no training data needed)
# ─────────────────────────────────────────────────────────────────────────────
import torch, dgl, warnings
warnings.filterwarnings("ignore")

from utils.utils import load_config
from models.model import DTIPredictor

configs = load_config(CONFIG_PATH)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = DTIPredictor(param=configs).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ DTIPredictor built. Trainable params: {n_params:,}")

# Synthetic forward pass
B, N_lig, N_pock = 2, 10, 15
g1 = dgl.batch([dgl.graph(([], [])) for _ in range(B)])
g1.add_nodes(N_lig * B)
g1.ndata['h']      = torch.randn(N_lig * B, 65).to(device)
g1.ndata['qm_mol'] = torch.randn(N_lig * B, 7).to(device)
g1.edata['e']      = torch.zeros(0, 21).to(device)

g2 = dgl.batch([dgl.graph(([], [])) for _ in range(B)])
g2.add_nodes(N_pock * B)
g2.ndata['h'] = torch.randn(N_pock * B, 40).to(device)
g2.edata['e'] = torch.zeros(0, 21).to(device)

g3 = dgl.batch([dgl.graph(([], [])) for _ in range(B)])
g3.add_nodes((N_lig + N_pock) * B)
g3.edata['e'] = torch.zeros(0, 1).to(device)

res = torch.randn(B, configs["prot_hidden_dim"] * 2).to(device)

model.eval()
with torch.no_grad():
    out = model(g1, g2, g3, res)
print(f"✅ Forward pass OK. Output shape: {out.shape}   (expected: [{B}, 1])")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Patch train.py to initialise QM loader
# ─────────────────────────────────────────────────────────────────────────────
# The original train.py doesn't call init_qm_loader().
# We patch it here programmatically so you don't have to edit the file.

TRAIN_PATH = os.path.join(SRC_DIR, "trainer", "train.py")
with open(TRAIN_PATH, "r") as f:
    train_src = f.read()

# num_process fix for Kaggle (limited CPU cores)
train_src = train_src.replace('num_process = 48', 'num_process = 4')

QM_PATCH = """
# ── MISATO QM initialisation (auto-patched by Kaggle notebook) ────────────
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.graph_constructor import init_qm_loader as _init_qm
_qm_path = configs.get('qm_hdf5_path', None)
if _qm_path and os.path.exists(_qm_path):
    _init_qm(_qm_path)
    print(f"[QM] Loaded: {_qm_path}")
else:
    print(f"[QM] ⚠️  HDF5 not found at '{_qm_path}'. Using zeros.")
# ─────────────────────────────────────────────────────────────────────────────
"""

# Insert the patch right after "configs = load_config(args.model_config_path)"
ANCHOR = "configs = load_config(args.model_config_path)"
if QM_PATCH.strip() not in train_src:
    patched = train_src.replace(ANCHOR, ANCHOR + "\n" + QM_PATCH, 1)
    with open(TRAIN_PATH, "w") as f:
        f.write(patched)
    print("✅ train.py patched with QM loader init + num_process=4.")
else:
    print("ℹ️  train.py already patched.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Launch training
# ─────────────────────────────────────────────────────────────────────────────
# This runs train.py as a subprocess so its output streams to the notebook.
# Training uses early stopping (patience=100), so it will stop before 1000 epochs
# if validation RMSE stops improving.

import subprocess, threading, time

os.makedirs(OUTPUT_DIR + "/model_save", exist_ok=True)

cmd = [
    sys.executable,
    os.path.join(SRC_DIR, "trainer", "train.py"),
    "--model_config_path", CONFIG_PATH,
]

if "RESUME_CKPT_PATH" in globals() and RESUME_CKPT_PATH.strip():
    cmd.extend(["--dti_ckpt_path", RESUME_CKPT_PATH.strip()])
    print(f"🔗 Configured to resume from checkpoint: {RESUME_CKPT_PATH}")

print("🚀 Starting training...")
print(f"   Command: {' '.join(cmd)}")
print(f"   Working dir: {SRC_DIR}")
print("   (output will stream below)\n")
print("─" * 70)

proc = subprocess.Popen(
    cmd,
    cwd=SRC_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

# Stream output line by line
for line in proc.stdout:
    print(line, end="", flush=True)

proc.wait()
print("─" * 70)
if proc.returncode == 0:
    print("\n✅ Training completed successfully!")
else:
    print(f"\n❌ Training exited with code {proc.returncode}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Find and display best model metrics
# ─────────────────────────────────────────────────────────────────────────────
import glob, pandas as pd

# Find the latest stats CSV
stat_files = sorted(glob.glob(os.path.join(SRC_DIR, "stats", "*_metrics.csv")))
if stat_files:
    latest = stat_files[-1]
    df = pd.read_csv(latest)
    print(f"Results from: {latest}\n")
    print(df.to_string(index=False))
else:
    print("No metrics CSV found yet. Check if training completed.")

# List saved model files
print("\nSaved model files:")
for f in glob.glob(OUTPUT_DIR + "/model_save/**/*", recursive=True):
    print(f"  {f}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Save outputs for download
# ─────────────────────────────────────────────────────────────────────────────
import zipfile, datetime

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
zip_path = f"/kaggle/working/mlpla_qm_results_{ts}.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    # Model checkpoints
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            full = os.path.join(root, file)
            arc  = os.path.relpath(full, OUTPUT_DIR)
            zf.write(full, arc)
    # Stats CSVs
    for f in glob.glob(os.path.join(SRC_DIR, "stats", "*.csv")):
        zf.write(f, os.path.basename(f))
    # Config used
    zf.write(CONFIG_PATH, "config_used.yaml")

size_mb = os.path.getsize(zip_path) / 1e6
print(f"✅ Results zipped to: {zip_path}  ({size_mb:.1f} MB)")
print("   Download via: File → right-click → Download")
