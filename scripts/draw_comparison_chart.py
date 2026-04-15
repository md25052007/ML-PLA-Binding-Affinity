import matplotlib.pyplot as plt
import numpy as np

# Data Extraction
metrics = ['RMSE ↓', 'MAE ↓', 'R² ↑', 'Pearson R ↑', 'C-Index ↑']
baseline = [1.3196, 0.9955, 0.6304, 0.7976, 0.8018]
qm_enhanced = [1.2964, 0.9984, 0.6274, 0.8325, 0.8119]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

# Create bars
rects1 = ax.bar(x - width/2, baseline, width, label='Baseline', color='#94a3b8', edgecolor='white', alpha=0.8)
rects2 = ax.bar(x + width/2, qm_enhanced, width, label='QM-Enhanced', color='#3b82f6', edgecolor='white')

# Add labels and styling
ax.set_ylabel('Score / Error Magnitude', fontsize=12)
ax.set_title('Model Performance Comparison: Baseline vs. QM-Enhanced', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# Save output
out_png = r"c:\Users\donth\Desktop\ds-project\ML-PLA-Binding-Affinity\results\casf2016\model_comparison_bar_chart.png"
plt.savefig(out_png, dpi=300)
print(f"Comparison chart saved to: {out_png}")
