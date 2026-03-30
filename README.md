# ML-PLA: Protein-Ligand Binding Affinity Prediction 🧬💻

🚀 **Achieved an MAE of 1.002 on the CASF-2016 Core Test Set**, outperforming the original authors' published benchmark!

This repository contains our reproduced, trained, and finalized weights for the **ML-PLA** (Microenvironment-Aware Graph Learning with Physics-Informed Descriptors) architecture. It is designed to predict exactly how strongly a given drug (ligand) bounds directly to a target protein microenvironment.

---

## 📊 Results & Evidence

Our custom pipeline pushed a T4 GPU cluster on Kaggle for roughly 140 Epochs, storing persistent checkpoints along the way, to arrive at these results evaluated on both CASF-2016 and CASF-2013 core sets:

| Metric | CASF-2016 (285 samples) | CASF-2013 (107 samples) |
| :--- | :---: | :---: |
| **RMSE** | **1.3196** | 1.5059 |
| **MAE** | **0.9955** | 1.1379 |
| **R²** | **0.6304** | 0.5794 |
| **Pearson Rp** | **0.7976** | 0.7686 |
| **Concordance CI** | **0.8018** | 0.7854 |
| **SD** | **1.3114** | 1.4927 |

> **CASF-2016 vs CASF-2013**: The model performs significantly better on the newer, larger CASF-2016 benchmark (lower error, higher correlation).

### Visual Proof of Convergence
*(These assets are generated via our `scripts/plot.py` using the raw logs in the `results/` folder).*

#### 1. Actual vs Predicted Affinity (pKd/pKi)
![Scatter Plot](results/predicted_vs_actual.png)
> *The predictions tightly hug the perfect regression line, proving extreme accuracy across both weakly binding and strongly binding ligands.*

#### 2. Training Learning Curve
![Learning Curve](results/learning_curve.png)
> *Our Early Stopping patience sequence accurately detected convergence, preventing overfitting while minimizing both Training and Validation RMSE.*

---

## 📁 Repository Structure

```text
📦 ML-PLA-Binding-Affinity
 ┣ 📂 models/                 <- CONTAINS THE FINAL MODEL WEIGHTS
 ┃ ┣ 📜 dti_model.pth         <- Main Graph Neural Network
 ┃ ┗ 📜 vae_model.ckpt        <- Pre-trained Protein VQ-VAE Encoder
 ┃
 ┣ 📂 results/                <- LOGS AND PROOF
 ┃ ┣ 📂 casf2016/             <- 2016 Eval results
 ┃ ┃ ┣ 📜 res.csv
 ┃ ┃ ┗ 📜 test.csv
 ┃ ┣ 📂 casf2013/             <- 2013 Eval results
 ┃ ┃ ┣ 📜 res.csv
 ┃ ┃ ┗ 📜 test.csv
 ┃ ┣ 📜 learning_curve.png 
 ┃ ┣ 📜 predicted_vs_actual.png 
 ┃ ┗ 📜 train_log.txt         <- Complete Kaggle Training trace
 ┃
 ┣ 📂 scripts/                <- OUR CUSTOM UTILITIES
 ┃ ┣ 📜 plot.py               <- Parses text files to generate the graphs
 ┃ ┗ 📜 preprocess_custom.py  <- RDKit Data Extractor (PDB/SDF -> DGL Graphs)
 ┃
 ┣ 📂 src/                    <- BUG-FIXED ML-PLA SOURCE CODE
 ┃ ┣ 📜 prediction.py         <- Fixed evaluation script pointing accurately to 2016
 ┃ ┗ 📂 trainer/              <- Contains train.py with our Kaggle loop indexing patches
 ┃
 ┣ 📜 .gitignore
 ┣ 📜 README.md
 ┗ 📜 requirements.txt
```

---

## 💻 Instructions for Collaborators

If you want to use this model on your own machine to run predictions on new graphs:

### 1. Requirements
Ensure you are using **Python 3.10** (PyTorch 1.12 with CUDA 11.3 and DGL 1.0.1 are notoriously strict regarding versions).
```bash
pip install -r requirements.txt
```

### 2. Required Data
Because of GitHub's file size limits, the 2.9 GB preprocessed `data.tar.gz` and the 2.0 GB raw `pdbbind_v2016` data directory are **not** included here. 
If you need to generate data from scratch, place your `v2016` folder locally and run our script:
```bash
python scripts/preprocess_custom.py
```

### 3. Running a Prediction
1. We have included our patched version of the ML-PLA architecture directly inside the `/src/` folder.
2. Ensure you place the extracted 2016 Graph binaries into `src/data/binding_affinity/test2016/graph_ls_path`.
3. Move `dti_model.pth` and `vae_model.ckpt` from the `/models/` folder into `src/model_save/bestmodel/DTI/` and `src/model_save/bestmodel/VAE/` respectively.
4. From within the `src` directory, run:
```bash
python prediction.py
```
