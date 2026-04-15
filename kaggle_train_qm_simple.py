# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ML-PLA + QM — Kaggle Training (Simple 4-Cell Version)                 ║
# ║  All NaN fixes are pre-baked into train.py and model.py                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── CELL 1: Install DGL ─────────────────────────────────────────────────
import os, sys
!pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html -q
!pip install dgllife rdkit prefetch_generator pyyaml h5py numba scipy scikit-learn tqdm -q
base = '/usr/local/lib/python3.12/dist-packages'
os.makedirs(f'{base}/torchdata/datapipes/iter', exist_ok=True)
os.makedirs(f'{base}/torchdata/dataloader2', exist_ok=True)
for f in ['torchdata/__init__.py','torchdata/datapipes/__init__.py',
           'torchdata/dataloader2/__init__.py','torchdata/dataloader2/graph.py']:
    open(f'{base}/{f}','w').write('')
with open(f'{base}/torchdata/datapipes/iter/__init__.py','w') as f:
    f.write('class IterDataPipe:\n def __iter__(self): return iter([])\n'
            'class IterableWrapper(IterDataPipe):\n def __init__(self,i=None,**k): pass\n def __iter__(self): return iter([])\n'
            'class Mapper(IterDataPipe): pass\nclass Filter(IterDataPipe): pass\nclass Batcher(IterDataPipe): pass\n')
gb = '/usr/local/lib/python3.12/dist-packages/dgl/graphbolt/__init__.py'
with open(gb,'r') as f: c = f.read()
c = c.replace('if not os.path.exists(path):\n        raise FileNotFoundError(\n            f"Cannot find DGL C++ graphbolt library at {path}"\n        )',
              'if not os.path.exists(path):\n        return')
with open(gb,'w') as f: f.write(c)
import dgl, torch; from rdkit import Chem
dgl.graph(([0,1],[1,2])).to('cuda')
print(f'✅ DGL: {dgl.__version__} | CUDA: {torch.cuda.is_available()} | rdkit: OK')


# ─── CELL 2: Paths + setup ───────────────────────────────────────────────
import shutil
QM_HDF5_PATH = '/kaggle/input/datasets/maithreyadonthi/misato-qm/QM.hdf5'
DATA_INPUT   = '/kaggle/input/datasets/maithreyadonthi/mlpla-preprocessed-data/data'
MLPLA_INPUT  = '/kaggle/input/datasets/maithreyadonthi/qm-integrated-folder/ML-PLA-Binding-Affinity'
WORK_DIR = '/kaggle/working/ML-PLA'
SRC_DIR  = f'{WORK_DIR}/src'
CONFIG_PATH = f'{SRC_DIR}/configs/config.yaml'
TRAIN_PATH  = f'{SRC_DIR}/trainer/train.py'
MODEL_SAVE  = f'{SRC_DIR}/model_save'
if os.path.exists(WORK_DIR): shutil.rmtree(WORK_DIR)
shutil.copytree(MLPLA_INPUT, WORK_DIR)
DATA_LINK = f'{SRC_DIR}/data'
if os.path.islink(DATA_LINK): os.unlink(DATA_LINK)
os.symlink(DATA_INPUT, DATA_LINK)
os.makedirs(MODEL_SAVE, exist_ok=True)
os.chdir(SRC_DIR)
sys.path.insert(0, SRC_DIR)
print(f'✅ Ready. cwd={os.getcwd()}')


# ─── CELL 3: Patch config + inject QM loader ─────────────────────────────
# NaN fixes (skip bad batches, NaN-safe metrics) are already in train.py
# LayerNorm on qm_mol_proj is already in model.py
import yaml, ast

# ── Config ────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
config['save_dir']         = MODEL_SAVE
config['qm_hdf5_path']     = QM_HDF5_PATH
config['use_qm_mol_feats'] = True
config['num_workers']      = 2
config['batch_size']       = 64
config['epoches']          = int(config.get('epochs', config.get('epoches', 1000)))
with open(CONFIG_PATH, 'w') as f:
    yaml.dump(config, f)
print(f"✅ config.yaml (epoches={config['epoches']}, batch={config['batch_size']})")

# ── train.py: inject QM loader + num_process fix ─────────────────────────
with open(TRAIN_PATH) as f:
    code = f.read()

code = code.replace('num_process = 48', 'num_process = 4')

ANCHOR = 'configs = load_config(args.model_config_path)'
INJECT = '\n'
INJECT += '    # ── QM loader ──────────────────────────────────────────────────\n'
INJECT += '    from dataset.graph_constructor import init_qm_loader as _init_qm\n'
INJECT += '    _qm_path = configs.get("qm_hdf5_path")\n'
INJECT += '    if _qm_path and os.path.exists(_qm_path):\n'
INJECT += '        _init_qm(_qm_path)\n'
INJECT += '        print("[QM] Loaded: " + str(_qm_path), flush=True)\n'
INJECT += '    else:\n'
INJECT += '        print("[QM] QM file not found", flush=True)\n'
INJECT += '    # ──────────────────────────────────────────────────────────────\n'
if '_init_qm' not in code:
    code = code.replace(ANCHOR, ANCHOR + INJECT, 1)
    print('✅ QM loader injected into train.py')
else:
    print('ℹ️  QM loader already present')

with open(TRAIN_PATH, 'w') as f:
    f.write(code)

try: ast.parse(code); print('✅ train.py syntax OK')
except SyntaxError as e: print(f'❌ train.py syntax error: {e}')

print('✅ All patches applied')


# ─── CELL 4: TRAIN ───────────────────────────────────────────────────────
os.chdir(SRC_DIR)
!python -u trainer/train.py --model_config_path {CONFIG_PATH}
