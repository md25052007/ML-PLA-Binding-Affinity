# ML-PLA: Complete Google Colab Notebook (Copy-Paste Ready)

> Open **colab.research.google.com** → New Notebook → Runtime → Change runtime type → **T4 GPU**

---

## CELL 1 — Verify GPU
```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
!nvidia-smi | head -15
```
✅ Should show T4 GPU. If not — go to Runtime → Change runtime type → GPU

---

## CELL 2 — Install Exact Dependencies (takes ~5 min)
```python
# These exact versions work with Python 3.10 + CUDA 11.x (Colab default)
!pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html -q

!pip install dgl==1.0.1+cu113 \
    -f https://data.dgl.ai/wheels/cu113/repo.html -q

!pip install dgllife==0.3.2 rdkit-pypi biopython \
    scipy scikit-learn prefetch_generator pyyaml tqdm numpy pandas -q

print("✅ All packages installed")
```

---

## CELL 3 — Verify Installation
```python
import torch, dgl
from dgllife.utils import BaseAtomFeaturizer
from rdkit import Chem
print("PyTorch:", torch.__version__)   # should be 1.12.1+cu113
print("DGL:", dgl.__version__)          # should be 1.0.1+cu113
print("CUDA:", torch.cuda.is_available())
print("✅ All imports OK")
```

---

## CELL 4 — Mount Google Drive (to save your model)
```python
from google.colab import drive
drive.mount('/content/drive')

import os
save_dir = '/content/drive/MyDrive/MLPLA_results'
os.makedirs(save_dir, exist_ok=True)
print("✅ Google Drive mounted. Results will be saved to:", save_dir)
```

---

## CELL 5 — Clone ML-PLA Repo
```python
import os

!git clone https://github.com/Biowust/ML-PLA.git -q
%cd ML-PLA
print("✅ Cloned. Current directory:")
!ls
```

---

## CELL 6 — Download Preprocessed Data from Zenodo (~3 min)
```python
import os

os.makedirs('./data', exist_ok=True)

print("Downloading preprocessed data (2.9 GB)...")
!wget "https://zenodo.org/records/15321538/files/data.tar.gz?download=1" \
     -O data.tar.gz -q --show-progress

print("Extracting...")
!tar -xzf data.tar.gz 2>/dev/null   # suppress mac metadata warnings

print("Cleaning up tar.gz...")
!rm -f data.tar.gz

print("✅ Data ready:")
!ls ./data/
!ls ./data/binding_affinity/
```

---

## CELL 7 — Check Config and Update Save Path
```python
import yaml, shutil

# Read config
with open('./configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Current config:")
print(f"  epochs: {config['epoches']}")
print(f"  batch_size: {config['batch_size']}")
print(f"  pre_epoch (VQ-VAE): {config['pre_epoch']}")
print(f"  save_dir: {config['save_dir']}")
print(f"  learning_rate: {config['lr']}")
print(f"  patience: {config['patience']}")
```

---

## CELL 8 — Look at Training Script to Know Paths Needed
```python
# Check what paths train.py expects
!grep -n "data\|path\|dir" trainer/train.py | head -30
```

---

## CELL 9 — Run Training (VQ-VAE pretraining + Full model)
```python
# This runs both the VQ-VAE pretraining AND the full ML-PLA model training
# Expected time: ~17 min (VQ-VAE) + ~1-2 hours (full model)
!python trainer/train.py 2>&1 | tee /content/drive/MyDrive/MLPLA_results/train_log.txt
```

---

## CELL 10 — Monitor Progress (run in a NEW cell while Cell 9 is running)
```python
# Run this in a separate cell to check progress without stopping training
!tail -50 /content/drive/MyDrive/MLPLA_results/train_log.txt
```

---

## CELL 11 — Evaluate on Test Set (after training completes)
```python
!python prediction.py 2>&1 | tee /content/drive/MyDrive/MLPLA_results/eval_log.txt
```

---

## CELL 12 — Compute Metrics and Show Results
```python
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle, os

# Load predictions (adjust path based on where prediction.py saves results)
result_dir = './result_best'
print("Result files:", os.listdir(result_dir))
```

---

## CELL 13 — Copy Trained Model to Google Drive
```python
import shutil, os, glob

# Copy all saved model files to Drive
model_files = glob.glob('./model_save/**/*.ckpt', recursive=True)
for f in model_files:
    dst = f'/content/drive/MyDrive/MLPLA_results/{os.path.basename(f)}'
    shutil.copy(f, dst)
    print(f"Saved: {dst}")

print(f"✅ {len(model_files)} model files saved to Google Drive")
```

---

## CELL 14 — Plot Results (Scatter + Training Loss)
```python
import matplotlib.pyplot as plt
import numpy as np

# Load training loss from log
losses = []
with open('/content/drive/MyDrive/MLPLA_results/train_log.txt') as f:
    for line in f:
        if 'loss:' in line:
            try:
                loss = float(line.strip().split('loss:')[-1].strip())
                losses.append(loss)
            except:
                pass

plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('ML-PLA Training Loss Curve')
plt.grid(True)
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/MLPLA_results/loss_curve.png', dpi=150)
plt.show()
print("✅ Loss curve saved")
```

---

## ⚠️ Important Notes

1. **Keep the Colab tab open** — if you close it, the session dies
2. **Cell 9** (training) will run for ~1.5–2 hours — just let it go
3. **Google Drive** saves everything permanently — even if Colab disconnects, your results are safe
4. If session disconnects mid-training → re-run Cells 1–6 (setup), then re-check if `./model_save/` has a checkpoint, then resume
5. **Session time**: Free Colab gives 12 hours — more than enough

---

## Session Disconnect Recovery

If Colab disconnects during training:
```python
# Check if training was saved
import glob
checkpoints = glob.glob('./model_save/**/*.ckpt', recursive=True)
print("Saved checkpoints:", checkpoints)
# If files exist, training saved progress — modify train.py to load from checkpoint
```

---

## After Training — Key Metrics to Report

```python
# Expected results from paper:
# Pearson R on CASF-2016: 0.845
# RMSE: 1.290
# MAE: 1.023
# 
# Your results should be close to these
# Run prediction.py and compare
```
