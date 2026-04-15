# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — Install deps + Clone ML-PLA source code
# ═══════════════════════════════════════════════════════════════════════════
import subprocess, os

# Install required packages (matching the training environment)
subprocess.run([
    "pip", "install", "-q", 
    "dgl", "-f", "https://data.dgl.ai/wheels/cu121/repo.html",
    "dgllife", "rdkit", "prefetch_generator", "numba", "pyyaml"
], check=True)

# Patch DGL backend issues for Kaggle Python 3.12+
base = '/usr/local/lib/python3.12/dist-packages'
if os.path.exists(base):     
    os.makedirs(f'{base}/torchdata/datapipes/iter', exist_ok=True)
    os.makedirs(f'{base}/torchdata/dataloader2', exist_ok=True)
    for f in ['torchdata/__init__.py','torchdata/datapipes/__init__.py','torchdata/dataloader2/__init__.py','torchdata/dataloader2/graph.py']:
        with open(f'{base}/{f}','w') as file: file.write('')
    with open(f'{base}/torchdata/datapipes/iter/__init__.py','w') as file:
        file.write('class IterDataPipe:\n def __iter__(self): return iter([])\nclass IterableWrapper(IterDataPipe):\n def __init__(self,i=None,**k): pass\n def __iter__(self): return iter([])\nclass Mapper(IterDataPipe): pass\nclass Filter(IterDataPipe): pass\nclass Batcher(IterDataPipe): pass\n')
    gb = f'{base}/dgl/graphbolt/__init__.py'
    if os.path.exists(gb):
        with open(gb,'r') as file: c = file.read()
        c = c.replace('if not os.path.exists(path):\n        raise FileNotFoundError(\n            f"Cannot find DGL C++ graphbolt library at {path}"\n        )','if not os.path.exists(path):\n        return')
        with open(gb,'w') as file: file.write(c)

print("✅ Dependencies installed and DGL patched.")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — Run Prediction on CASF-2016 Test Set
# ═══════════════════════════════════════════════════════════════════════════
import sys, glob, torch, warnings, pickle
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from numba import njit
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

# ── Paths — everything comes from the bundled zip datasets ──────────────
# Auto-detect the extraction path (Kaggle might extract with or without parent folder)
input_matches = sorted(glob.glob("/kaggle/input/**/ckpt_ep340.pt", recursive=True))
assert input_matches, "❌ ckpt_ep340.pt not found! Make sure you uploaded the Inference Zip dataset."
INPUT_BASE = os.path.dirname(input_matches[0])
print(f"✅ Auto-detected input base: {INPUT_BASE}")

# ── Add source code to path ───────────────────────────────────────────────
# Using the uploaded zip dataset source code
sys.path.insert(0, f"{INPUT_BASE}/src")

from dataset.graph_constructor import *
from utils.utils import *
from models.model import *
from models.codebook import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

dti_path = CKPT_340_PATH

# Find the QM dataset (it contains the atom-level features we need to inject)
qm_matches = sorted(glob.glob("/kaggle/input/**/QM*.hdf5", recursive=True))
assert qm_matches, "❌ QM.hdf5 not found! Ensure you added the Misato QM dataset to your Kaggle notebook."
QM_DB_PATH = qm_matches[0]

print(f"\n✅ VAE  : {vae_path}")
print(f"✅ DTI  : {dti_path}")
print(f"✅ CFG  : {cfg_path}")
print(f"✅ QM DB: {QM_DB_PATH}")
print(f"✅ Will use epoch 340 checkpoint for inference.")

# Initialize the QM loader globally for the graph constructor tools
from dataset.graph_constructor import init_qm_loader
init_qm_loader(QM_DB_PATH)

# Test data DGL graphs
assert os.path.exists(TEST_GRAPH_PATH), f"❌ Test graph path {TEST_GRAPH_PATH} not found!"
print(f"✅ TEST GRAPHS : {TEST_GRAPH_PATH}")

# ── Load config ───────────────────────────────────────────────────────────
configs = load_config(cfg_path)
set_random_seed(configs['seed'])

# ── CI metric ─────────────────────────────────────────────────────────────
@njit
def CI(y_true, y_pred):
    summ = 0; pair = 0
    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1*(y_pred[i]>y_pred[j]) + 0.5*(y_pred[i]==y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1*(y_pred[i]<y_pred[j]) + 0.5*(y_pred[i]==y_pred[j])
            else:
                pair -= 1
    return summ/pair if pair != 0 else 0

# ── Dataset ───────────────────────────────────────────────────────────────
class DataLoaderX(DataLoader):
    def __iter__(self): return BackgroundGenerator(super().__iter__())

class GraphSet(object):
    def __init__(self, path):
        print('Loading previously saved dgl graphs and corresponding data...')
        b = lambda f: open(os.path.join(path, f), 'rb')
        self.graphs1       = pickle.load(b('g1.bin'))
        self.graphs2       = pickle.load(b('g2.bin'))
        self.graphs3       = pickle.load(b('g3.bin'))
        self.residue_graph = pickle.load(b('residue.bin'))
        self.keys          = pickle.load(b('keys.bin'))
        self.labels        = pickle.load(b('labels.bin'))
    def __len__(self): return len(self.labels)
    def __getitem__(self, indx):
        g1 = self.graphs1[indx]
        key = self.keys[indx]
        
        # ── Runtime QM injection for pre-built graphs ──
        # This converts the 40-dim baseline graphs into 65-dim QM graphs on the fly
        from dataset.graph_constructor import _QM_LOADER, _get_qm_atom_feats, _get_qm_mol_feats, QM_ATOM_FEAT_DIM
        if _QM_LOADER is not None:
            h = g1.ndata.get('h')
            # Only inject if not already done (shape check)
            if h is not None and h.shape[1] < 40 + QM_ATOM_FEAT_DIM:
                qm_a = _get_qm_atom_feats(key, g1.num_nodes()) 
                g1.ndata['h'] = torch.cat([h, qm_a], dim=1) # Now 65 dims!
            if 'qm_mol' not in g1.ndata:
                qm_m = _get_qm_mol_feats(key)              
                g1.ndata['qm_mol'] = qm_m.unsqueeze(0).expand(g1.num_nodes(), -1)

        return (g1, self.graphs2[indx], self.graphs3[indx],
                self.residue_graph[indx],
                torch.tensor(self.labels[indx], dtype=torch.float),
                key)

print("\n🚀 Running Final Evaluation on CASF-2016 Test Set...")
test_dataset = GraphSet(TEST_GRAPH_PATH)
print(f"the number of test data: {len(test_dataset)}")

# ── Load VAE ──────────────────────────────────────────────────────────────
vae_model = CodeBook(configs).to(device)
vae_model.load_state_dict(torch.load(vae_path, map_location=device))
vae_model.eval()
print("✅ VAE model loaded.")

# ── Load DTI ──────────────────────────────────────────────────────────────
DTIModel = DTIPredictor(param=configs).to(device)
n_params = sum(p.numel() for p in DTIModel.parameters() if p.requires_grad)
print(f"number of parameters : {n_params}")

# Handle both nested dict and raw state_dict dynamically
state_dict = torch.load(dti_path, map_location=device)
if 'model_state_dict' in state_dict:
    DTIModel.load_state_dict(state_dict['model_state_dict'])
else:
    DTIModel.load_state_dict(state_dict)

DTIModel.eval()
print("✅ DTI model loaded.")

# ── Inference ─────────────────────────────────────────────────────────────
loader = DataLoaderX(
    test_dataset,
    batch_size=configs['batch_size'],
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn_v2,
    drop_last=False
)

true_all, pred_all, key_all = [], [], []
with torch.no_grad():
    for bg1, bg2, bg3, bg4, Ys, keys in loader:
        bg1,bg2,bg3,bg4,Ys = [x.to(device) for x in [bg1,bg2,bg3,bg4,Ys]]
        prot_embed = vae_model.Protein_Encoder.forward(vae_model.vq_layer, bg4).to(device)
        out = DTIModel(bg1, bg2, bg3, prot_embed)
        true_all.append(Ys.cpu().numpy())
        pred_all.append(out.cpu().numpy())
        key_all.append(keys)

test_true = np.concatenate([np.array(s).flatten() for s in true_all])
test_pred = np.concatenate([np.array(s).flatten() for s in pred_all])
test_keys = np.concatenate([np.array(s).flatten() for s in key_all])

# ── Save test.csv ─────────────────────────────────────────────────────────
os.makedirs('/kaggle/working/result', exist_ok=True)
pd.DataFrame({'key': test_keys, 'test_true': test_true, 'test_pred': test_pred})\
  .to_csv('/kaggle/working/result/test.csv', index=False)

# ── Compute all metrics ───────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(test_true, test_pred))
mae  = mean_absolute_error(test_true, test_pred)
r2   = r2_score(test_true, test_pred)
rp   = pearsonr(test_true, test_pred)[0]
lr   = LinearRegression()
lr.fit([[p] for p in test_pred], test_true)
sd   = (((test_true - lr.predict([[p] for p in test_pred]))**2).sum() / (len(test_true)-1))**0.5
ci   = CI(test_true.astype(np.float64), test_pred.astype(np.float64))

# ── Print results in requested format ────────────────────────────────────
print("\n***best model***")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"Pearson correlation coefficient: {rp}")
print(f"Standard Deviation: {sd}")
print(f"R2 score: {r2}")
print(f"CI score: {ci}")
print(f"\ntest_rmse:{rmse:.4f} \t test_r2:{r2:.4f} \t test_mae:{mae:.4f} \t test_rp:{rp:.4f}\t test_ci:{ci:.4f}\t test_sd:{sd:.4f}")

# ── Save res.csv ──────────────────────────────────────────────────────────
pd.DataFrame(
    [['test', rmse, r2, mae, rp, ci, sd]],
    columns=['group','rmse','r2','mae','rp','ci','sd']
).to_csv('/kaggle/working/result/res.csv', index=False)

print("\n✅ Saved to /kaggle/working/result/test.csv and res.csv")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — Display results table
# ═══════════════════════════════════════════════════════════════════════════
import pandas as pd

print("=== Prediction Results (first 10) ===")
df = pd.read_csv('/kaggle/working/result/test.csv')
print(df.head(10).to_string(index=False))

print("\n=== Final Metrics ===")
res = pd.read_csv('/kaggle/working/result/res.csv')
print(res.to_string(index=False))
