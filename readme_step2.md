# Step 2 Execution: Resuming Training from Epoch 340

That's excellent news! Since you have the checkpoint from Epoch 340, we won't lose your 12 hours of compute time. 

I have modified the main Kaggle script (`kaggle_train_qm.py`) to easily digest your checkpoint.

## 🏃‍♂️ How to Execute Step 2 in Kaggle

Follow these steps exactly in your Kaggle Notebook to resume training:

### 1. Upload your Checkpoint
1. In your Kaggle notebook, click **Add Data** (top right corner).
2. Click **Upload** and upload the `dti_model.pth` file (the epoch 340 checkpoint you downloaded).
3. Give the dataset a name, e.g., `mlpla-checkpoint-340`.

### 2. Update the Kaggle Script
In the first few cells of your Kaggle notebook (specifically **CELL 3 — Set up paths**), I have added a new variable `RESUME_CKPT_PATH`. 

Update it to point to your newly uploaded dataset. It should look something like this:

```python
# ── Adjust these to match YOUR Kaggle dataset folder names ──────────────────
MLPLA_INPUT     = "/kaggle/input/ml-pla-binding-affinity"   # your code dataset
QM_HDF5_PATH    = "/kaggle/input/misato-qm/QM (1).hdf5"     # QM HDF5 dataset
BINDING_DATA    = "/kaggle/input/pdbbind-casf2016"          # PDBbind data

# Optional: Resume training from a previous checkpoint (e.g. after a 12h timeout)
# Update this line to match the dataset you just uploaded!
RESUME_CKPT_PATH = "/kaggle/input/mlpla-checkpoint-340/dti_model.pth" 
```

### 3. Run the Notebook
Go to the top menu and select **Run All**. 

**What will happen under the hood?**
- The script (`kaggle_train_qm.py` Cell 10) will automatically detect that `RESUME_CKPT_PATH` is set.
- It will pass `--dti_ckpt_path` to `train.py`.
- `train.py` will initialise the model and then **load your epoch 340 weights**.
- It will then begin training at the next epoch, picking up exactly where it left off!

## ✅ Verifying it Worked
When you reach Cell 10 in Kaggle and the training begins, watch the console logs. 
Before the first epoch starts, you should see this output:

```text
🔗 Configured to resume from checkpoint: /kaggle/input/mlpla-checkpoint-340/dti_model.pth
...
Loading previous DTI checkpoint from: /kaggle/input/mlpla-checkpoint-340/dti_model.pth
```

If you see that, you've successfully restored your progress! Let me know if you run into any issues during the upload or execution.
