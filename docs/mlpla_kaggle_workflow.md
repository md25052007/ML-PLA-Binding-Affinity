# ML-PLA: Complete Step-by-Step Kaggle Workflow
> **For**: Team-3 (Maharshi, Gokul, Maithreya) — NIT Karnataka  
> **Platform**: Kaggle Notebooks (Free GPU)  
> **Goal**: Reproduce ML-PLA results + MISATO extension  
> **Difficulty**: Beginner-friendly — every single step is listed

---

## 🗺️ Overview of All Phases

| Phase | What You Do |
|---|---|
| **0** | Create Kaggle account & enable GPU |
| **1** | Download datasets (PDBBind + preprocessed data) |
| **2** | Clone ML-PLA GitHub repo inside Kaggle |
| **3** | Install all dependencies |
| **4** | Run data preprocessing |
| **5** | Pretrain the VQ-VAE codebook |
| **6** | Train the full ML-PLA model |
| **7** | Evaluate on test sets |
| **8** | MISATO extension (QM + MD features) |
| **9** | Generate visualizations |

---

## ✅ PHASE 0 — Kaggle Account & Notebook Setup

### Step 0.1 — Create a Kaggle Account
1. Go to [https://www.kaggle.com](https://www.kaggle.com)
2. Click **Register** → sign up with Google or email
3. Verify your phone number (required to use GPU — it's free, just for identity verification)
   - Go to **Profile → Settings → Phone Verification**

### Step 0.2 — Create a New Notebook
1. On the Kaggle homepage, click **"Create"** → **"New Notebook"**
2. A Jupyter-like environment opens in your browser
3. At the top, rename it: click the title `notebook1` → type `mlpla-run`

### Step 0.3 — Enable the GPU
1. On the **right sidebar**, find the **"Session Options"** panel
2. Under **Accelerator**, select **GPU T4 x2** (or just **GPU T4** if that's the only option)
3. Click **"Save"** — the session will restart with GPU enabled
4. Verify GPU is active by running this in a cell:
```python
!nvidia-smi
```
You should see a table showing `Tesla T4` and `16160 MiB` of memory.

### Step 0.4 — Set Notebook to "No Internet" → Switch to "Internet ON"
1. In the right sidebar, find **"Internet"** toggle
2. Make sure it is **ON** (you need it to clone GitHub and download data)

---

## ✅ PHASE 1 — Download the Datasets

### Step 1.1 — Download Preprocessed Data from Zenodo (RECOMMENDED — Use This First)
> The authors already preprocessed the PDBBind data and uploaded it. This saves hours of preprocessing time.

In a new cell, run:
```python
# Download preprocessed data from Zenodo
!pip install zenodo-get -q
!zenodo_get 15321538
```
> If `zenodo_get` doesn't work, use `wget` directly:
```python
!wget -O preprocessed_data.zip "https://zenodo.org/records/15321538/files/data.zip"
!unzip preprocessed_data.zip -d ./data/
```
Check what was downloaded:
```python
!ls ./data/
```

### Step 1.2 — Understand the Data Folder Structure
After downloading, your `./data/` folder should look like:
```
data/
├── train.pkl       ← preprocessed training graphs (11,904 samples)
├── val.pkl         ← validation set (1,000 samples)
├── test_2016.pkl   ← CASF-2016 test set (285 samples)
├── test_2013.pkl   ← CASF-2013 test set (107 samples)
└── affinity.csv    ← binding affinity labels
```
> If the filenames differ, run `!ls -lh ./data/` and note the actual names — you may need to adjust paths later.

### Step 1.3 — (Optional) Download Raw PDBBind Data
> Skip this if you used the preprocessed Zenodo data above. Only do this if you want to run `preprocess.py` from scratch.

1. Go to [http://www.pdbbind.org.cn/](http://www.pdbbind.org.cn/)
2. Register for a free account
3. Download: **PDBBind v2016 General Set** (~7 GB)
4. Upload to Kaggle:
   - Kaggle Sidebar → **"Add Data"** → **"Upload"** → drag and drop the ZIP
   - Or: use Kaggle API (see Step 1.4)

### Step 1.4 — (Optional) Download MISATO Dataset for Phase 8
```python
# MISATO dataset for QM/MD features (used in the extension phase)
!wget -O misato.zip "https://zenodo.org/records/7711953/files/MISATO.zip"
!unzip misato.zip -d ./misato/
```

---

## ✅ PHASE 2 — Clone the ML-PLA Repository

### Step 2.1 — Clone from GitHub
Run in a new cell:
```bash
!git clone https://github.com/Biowust/ML-PLA.git
```

### Step 2.2 — Move Into the Project Directory
```bash
%cd ML-PLA
```
> Note: Use `%cd` (magic command) not `!cd` — the `!cd` command doesn't persist in Jupyter notebooks.

### Step 2.3 — Check the Folder Structure
```bash
!ls -lh
```
Expected output:
```
data/
model/
trainer/
preprocess.py
prediction.py
README.md
requirements.txt
...
```

### Step 2.4 — Copy the Downloaded Data Into the Repo
```python
!cp -r /kaggle/working/data ./data
```
> Or adjust the source path based on where Zenodo extracted the files.

---

## ✅ PHASE 3 — Install All Dependencies

> Run all of these in order in a single cell or separate cells.

### Step 3.1 — Install PyTorch (CUDA 11.3)
```bash
!pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html -q
```

### Step 3.2 — Install DGL (Deep Graph Library) with CUDA
```bash
!pip install dgl==1.0.1+cu113 \
  -f https://data.dgl.ai/wheels/cu113/repo.html -q
!pip install dgllife==0.3.2 -q
```

### Step 3.3 — Install Scientific Libraries
```bash
!pip install numpy pandas scikit-learn scipy -q
!pip install rdkit-pypi biopython -q
!pip install matplotlib seaborn tqdm -q
```

### Step 3.4 — Verify Everything Is Installed
```python
import torch
import dgl
import rdkit
import numpy as np

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("DGL version:", dgl.__version__)
print("✅ All dependencies OK")
```
Expected output:
```
PyTorch version: 1.12.1+cu113
CUDA available: True
GPU: Tesla T4
DGL version: 1.0.1+cu113
✅ All dependencies OK
```

---

## ✅ PHASE 4 — Data Preprocessing

> **Skip this phase if you used the Zenodo preprocessed data in Step 1.1.**  
> Only run this if you downloaded the raw PDBBind data.

### Step 4.1 — Run the Preprocessing Script
```bash
!python preprocess.py
```
This script will:
- Read all `.pdb` and `.mol2` files
- Extract atom features and residue features
- Build all 4 types of graphs (𝒢_r, 𝒢_p, 𝒢_l, 𝒢_pl)
- Save them as `.pkl` files in `./data/`

> ⏱️ **Expected time**: 2–4 hours for the full dataset. Use preprocessed data instead!

### Step 4.2 — Verify Preprocessing Output
```python
import pickle
with open('./data/train.pkl', 'rb') as f:
    train_data = pickle.load(f)
print(f"Training samples loaded: {len(train_data)}")
print(f"Sample keys: {train_data[0].keys()}")
```

---

## ✅ PHASE 5 — Pretrain the VQ-VAE Codebook

> This is a **separate pretraining step** that must run before the main model training.

### Step 5.1 — Find the Pretraining Script
```bash
!ls trainer/
```
Look for a file named something like `pretrain.py` or `pretrain_codebook.py`.

### Step 5.2 — Run Pretraining
```bash
!python trainer/pretrain.py \
  --data_path ./data/train.pkl \
  --epochs 50 \
  --save_path ./checkpoints/codebook.pt
```
> If the argument names differ, check the script: `!python trainer/pretrain.py --help`

### Step 5.3 — Monitor Pretraining Progress
You'll see output like:
```
Epoch 1/50 | Loss: 2.341 | Time: 19.6s
Epoch 2/50 | Loss: 2.108 | Time: 19.4s
...
Epoch 50/50 | Loss: 0.872 | Time: 19.5s
✅ Codebook saved to ./checkpoints/codebook.pt
```
> ⏱️ **Total time**: ~50 × 20s = ~17 minutes on T4 GPU.

### Step 5.4 — Verify the Codebook Was Saved
```python
import os
print("Codebook saved:", os.path.exists('./checkpoints/codebook.pt'))
```

---

## ✅ PHASE 6 — Train the Full ML-PLA Model

### Step 6.1 — Run the Training Script
```bash
!python trainer/train.py \
  --data_path ./data/ \
  --codebook_path ./checkpoints/codebook.pt \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.001 \
  --save_path ./checkpoints/mlpla_best.pt
```
> Again, check available args with: `!python trainer/train.py --help`

### Step 6.2 — What to Watch During Training
Each epoch will print something like:
```
Epoch 001 | Train Loss: 1.432 | Val RMSE: 1.523 | Val R: 0.712
Epoch 002 | Train Loss: 1.318 | Val RMSE: 1.441 | Val R: 0.741
...
🏆 Best model saved at epoch 087 (Val RMSE: 1.179)
```
The model with the **lowest validation RMSE** is saved automatically.

### Step 6.3 — Check Training Time
> ⏱️ **Rough estimate**: 200 epochs on T4 GPU with ~12K samples ≈ **1–3 hours**  
> Kaggle sessions last 12 hours, so this is fine.

### Step 6.4 — Save Your Notebook (IMPORTANT!)
Click **"Save Version"** at the top right while training runs. This commits the notebook so your output isn't lost if the session times out.

---

## ✅ PHASE 7 — Evaluate the Model

### Step 7.1 — Run Evaluation on CASF-2016
```bash
!python prediction.py \
  --model_path ./checkpoints/mlpla_best.pt \
  --test_data ./data/test_2016.pkl \
  --output ./results/casf2016_results.csv
```

### Step 7.2 — Run Evaluation on CASF-2013 (Optional)
```bash
!python prediction.py \
  --model_path ./checkpoints/mlpla_best.pt \
  --test_data ./data/test_2013.pkl \
  --output ./results/casf2013_results.csv
```

### Step 7.3 — Compute Metrics in Python
```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

results = pd.read_csv('./results/casf2016_results.csv')
y_true = results['true_affinity'].values
y_pred = results['predicted_affinity'].values

pearson_r, _ = pearsonr(y_true, y_pred)
spearman_rho, _ = spearmanr(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("=" * 40)
print(f"  Pearson R  : {pearson_r:.4f}   (target > 0.84)")
print(f"  Spearman ρ : {spearman_rho:.4f}")
print(f"  RMSE       : {rmse:.4f}   (target < 1.20)")
print(f"  MAE        : {mae:.4f}   (target < 0.90)")
print("=" * 40)
```

### Step 7.4 — Expected Results
| Metric | Your Result | Paper's ML-PLA |
|---|---|---|
| Pearson R | ~0.845 | 0.845 |
| RMSE | ~1.179 | 1.179 |
| MAE | ~0.892 | 0.892 |

---

## ✅ PHASE 8 — MISATO Extension (QM + MD Features)

### Step 8.1 — Load MISATO Data
```python
import h5py
import numpy as np

misato_file = './misato/MISATO.hdf5'
with h5py.File(misato_file, 'r') as f:
    print("Available keys:", list(f.keys())[:5])
```

### Step 8.2 — Extract QM and MD Features
```python
def load_misato_features(complex_id, misato_file):
    with h5py.File(misato_file, 'r') as f:
        if complex_id not in f:
            return None, None
        group = f[complex_id]
        qm_features = np.array(group['partial_charges'])       # shape: (n_atoms,)
        md_features  = np.array(group['trajectory_coordinates'])  # shape: (frames, n_atoms, 3)
        rmsf = md_features.std(axis=0).mean(axis=-1)           # RMSF per atom
    return qm_features, rmsf
```

### Step 8.3 — Modify the Preprocessing to Add MISATO Features
```python
# In your graph construction / preprocessing step:
for complex_id, graph_data in dataset.items():
    qm_feats, md_feats = load_misato_features(complex_id, misato_file)
    
    if qm_feats is not None:
        extra_features = np.column_stack([qm_feats, md_feats])
    else:
        # Zero-pad if complex not in MISATO
        n_atoms = graph_data['ligand_atoms']
        extra_features = np.zeros((n_atoms, 2))
    
    # Append to existing atom node features
    graph_data['atom_features'] = np.concatenate(
        [graph_data['atom_features'], extra_features], axis=1
    )
```

### Step 8.4 — Update Model Input Dimension
In the model code (likely `model/mlpla.py`), find the line that defines the input feature size:
```python
# Find something like:
self.atom_encoder = nn.Linear(original_feat_dim, hidden_dim)

# Change to:
self.atom_encoder = nn.Linear(original_feat_dim + 2, hidden_dim)  # +2 for QM + RMSF
```

### Step 8.5 — Train 4 Ablation Variants
```bash
# Variant 1: Baseline (no MISATO)
!python trainer/train.py --misato none --save_path ./checkpoints/v1_baseline.pt

# Variant 2: + QM features only
!python trainer/train.py --misato qm --save_path ./checkpoints/v2_qm.pt

# Variant 3: + MD features only
!python trainer/train.py --misato md --save_path ./checkpoints/v3_md.pt

# Variant 4: + QM + MD features
!python trainer/train.py --misato both --save_path ./checkpoints/v4_qm_md.pt
```
> You may need to add the `--misato` argument to the training script yourself.

---

## ✅ PHASE 9 — Visualizations

### Step 9.1 — Scatter Plot: Predicted vs True Affinity
```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.3, s=40)

# Regression line
m, b = np.polyfit(y_true, y_pred, 1)
x_line = np.linspace(y_true.min(), y_true.max(), 100)
ax.plot(x_line, m*x_line + b, 'r-', lw=2)

r, _ = pearsonr(y_true, y_pred)
ax.set_xlabel("Experimental Binding Affinity (pKa)", fontsize=13)
ax.set_ylabel("Predicted Binding Affinity (pKa)", fontsize=13)
ax.set_title(f"ML-PLA on CASF-2016  |  Pearson R = {r:.3f}", fontsize=14)
plt.tight_layout()
plt.savefig('./results/scatter_casf2016.png', dpi=150)
plt.show()
```

### Step 9.2 — Ablation Bar Chart
```python
models = ['Baseline', '+QM', '+MD', '+QM+MD']
r_values = [0.845, 0.851, 0.848, 0.857]    # replace with your actual values
rmse_values = [1.179, 1.152, 1.163, 1.138]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

axes[0].bar(models, r_values, color=colors)
axes[0].set_ylabel("Pearson R ↑", fontsize=13)
axes[0].set_title("MISATO Ablation — Pearson R", fontsize=13)
axes[0].set_ylim(0.82, 0.87)

axes[1].bar(models, rmse_values, color=colors)
axes[1].set_ylabel("RMSE ↓", fontsize=13)
axes[1].set_title("MISATO Ablation — RMSE", fontsize=13)
axes[1].set_ylim(1.10, 1.22)

plt.tight_layout()
plt.savefig('./results/ablation_chart.png', dpi=150)
plt.show()
```

### Step 9.3 — Training Loss Curve
```python
# Load your saved training logs (adjust filename as needed)
import pandas as pd
logs = pd.read_csv('./results/training_log.csv')

plt.figure(figsize=(9, 5))
plt.plot(logs['epoch'], logs['train_loss'], label='Train Loss', lw=2)
plt.plot(logs['epoch'], logs['val_rmse'], label='Val RMSE', lw=2, linestyle='--')
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Loss / RMSE", fontsize=13)
plt.title("ML-PLA Training Curves", fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('./results/training_curves.png', dpi=150)
plt.show()
```

---

## ✅ PHASE 10 — Save & Export Your Work

### Step 10.1 — Download Results from Kaggle
1. In the Kaggle notebook sidebar, click **"Output"**
2. Your files will appear under `/kaggle/working/`
3. Click on any file → **"Download"**

### Step 10.2 — Save the Notebook Version
1. Click **"Save Version"** (top right)
2. Choose **"Save & Run All (Commit)"**
3. This creates a permanent, reproducible snapshot of your run

### Step 10.3 — Share the Notebook (For Your Report)
1. Click **"Share"** → set to **Public**
2. Copy the link — you can include it in your report as a reproducible experiment link

---

## 📋 Final Checklist

| # | Step | Status |
|---|---|---|
| 0.1 | Created Kaggle account | ☐ |
| 0.2 | Phone verified (for GPU access) | ☐ |
| 0.3 | Created new notebook | ☐ |
| 0.4 | Enabled GPU T4 | ☐ |
| 1.1 | Downloaded preprocessed Zenodo data | ☐ |
| 1.4 | Downloaded MISATO dataset | ☐ |
| 2.1 | Cloned ML-PLA GitHub repo | ☐ |
| 3.x | Installed all dependencies (PyTorch, DGL, etc.) | ☐ |
| 3.4 | Verified GPU is working | ☐ |
| 5.x | Ran VQ-VAE pretraining (~17 min) | ☐ |
| 6.x | Trained full ML-PLA model (~1–3 hrs) | ☐ |
| 7.x | Evaluated on CASF-2016 | ☐ |
| 7.3 | Computed Pearson R, RMSE, MAE | ☐ |
| 8.x | Added MISATO QM + MD features | ☐ |
| 8.5 | Ran all 4 ablation variants | ☐ |
| 9.x | Generated all 3 plots | ☐ |
| 10.2 | Saved notebook version | ☐ |

---

## ⚠️ Common Issues & Fixes

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Reduce `--batch_size` from 32 to 16 |
| `ModuleNotFoundError: dgl` | Re-run the DGL install cell and restart the kernel |
| Session timed out | Click "Save Version" often; Kaggle saves output even if session dies |
| Zenodo download fails | Use `!wget` with direct file URL from the Zenodo page |
| `No such file: train.pkl` | Check `!ls /kaggle/working/` and adjust the data path |
| GPU not detected | Check Accelerator in sidebar; restart session after enabling |
