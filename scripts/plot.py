import pandas as pd
import matplotlib.pyplot as plt
import re
import os

df = pd.read_csv('test.csv')
plt.figure(figsize=(8,6))
plt.scatter(df['test_true'], df['test_pred'], alpha=0.6, edgecolors='w', color='#1f77b4', s=60)
plt.plot([2, 13], [2, 13], 'r--', lw=2, label='Perfect Prediction')
plt.title('CASF-2016 Prediction Summary: Actual vs Predicted pKd/pKi', fontsize=14)
plt.xlabel('Original / Actual Binding Affinity (pKd/pKi)', fontsize=12)
plt.ylabel('Predicted Binding Affinity (pKd/pKi)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=300)
plt.close()

epochs = []
train_rmse = []
val_rmse = []

with open('train_log.txt', 'r') as f:
    for line in f:
        if 'train_rmse:' in line and 'valid_rmse:' in line:
            m = re.search(r'epoch:(\d+)\s+train_rmse:([\d.]+)\s+valid_rmse:([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                train_rmse.append(float(m.group(2)))
                val_rmse.append(float(m.group(3)))

if epochs:
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_rmse, label='Train RMSE', color='#1f77b4', lw=2)
    plt.plot(epochs, val_rmse, label='Validation RMSE', color='#d62728', lw=2)
    plt.title('Training and Validation RMSE Learning Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300)
    plt.close()
