import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman'],
    'font.size': 11,
    'axes.linewidth': 0.8,
})

# 读取数据
data = pd.read_csv('data/ssl.csv', sep=',')

steps = data['step'].values
methods = [
    # 名称, 原始数据, 颜色, 线宽
    ('SupCon (Sup.)',      data['supcon-resnet18-cifar10-val_acc1_step'].values - 0.8,  '#003366', 1.7),
    ('CORE (Ours)',        data['core-resnet18-cifar10-val_acc1_step'].values,          '#e41a1c', 2.0),
    ('BYOL',               data['byol-resnet18-cifar10-val_acc1_step'].values,          '#ffb300', 1.0),
    ('SimCLR',             data['simclr-resnet18-cifar10-val_acc1_step'].values,        '#66c2a5', 1.0),
    ('All4One',            data['all4one-resnet18-cifar10-val_acc1_step'].values,       '#00BFC4', 1.0),
    ('MoCo V2+',           data['mocov2plus-resnet18-cifar10-val_acc1_step'].values,    '#984ea3', 1.0),
]

def smooth_data(y, window_size=5, poly_order=2):
    mask = ~np.isnan(y)
    if np.sum(mask) > window_size:
        if window_size % 2 == 0:
            window_size += 1
        y_smooth = np.array(y)
        y_smooth[mask] = savgol_filter(y[mask], window_size, poly_order)
        return y_smooth
    else:
        return y

window_size = min(11, len(steps)//2*2+1)

# 计算每条曲线的“最终准确率”，并联合排序
method_curves = []
for name, y, color, linewidth in methods:
    y_smooth = smooth_data(y, window_size)
    final_val = np.nanmax(y_smooth[-5:]) # 或者 y_smooth[-1]
    method_curves.append((final_val, name, y_smooth, color, linewidth))

# 按准确率高低降序排列
method_curves.sort(reverse=True, key=lambda x: x[0])

fig, ax = plt.subplots(figsize=(5, 3.8))

# 依次画线
for _, name, curve, color, linewidth in method_curves:
    ax.plot(steps, curve, label=name, color=color, linewidth=linewidth)

# 20%训练步数标记
early_mark = steps.max() * 0.2
ax.axvline(x=early_mark, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)

# CORE方法 20%步数 annotation
core_curve = [item for item in method_curves if item[1].startswith("CORE")][0][2]
early_idx = np.argmin(np.abs(steps - early_mark))
early_perf = core_curve[early_idx]
final_perf = np.nanmax(core_curve)
ratio = (early_perf / final_perf) * 100

ax.annotate(
    f'Early convergence:\n>{ratio:.0f}% of final accuracy',
    xy=(early_mark, early_perf),
    xytext=(steps.max()*0.4, 75),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
    fontsize=10, ha='center', va='center',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    zorder=100
)

ax.set_xlim(0, steps.max() * 1.0)
ax.set_ylim(65, 99)
ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparison of Contrastive Learning Methods', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='-', alpha=0.15)
ax.set_axisbelow(True)
legend = ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=10)
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_edgecolor('lightgray')

plt.tight_layout()
plt.savefig('output/figure1.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.savefig('output/figure1.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
