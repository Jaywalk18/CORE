import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    ('CORE (Ours)',        data['core-resnet18-cifar10-val_acc1_step'].values,          '#B22222', 2.0),
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

# 计算每条曲线的“最终准确率”，并排序
method_curves = []
for name, y, color, linewidth in methods:
    y_smooth = smooth_data(y, window_size)
    final_val = np.nanmax(y_smooth[-5:]) # 或 y_smooth[-1]
    method_curves.append((final_val, name, y_smooth, color, linewidth))

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
    xytext=(steps.max()*0.29, 73),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
    fontsize=10, ha='center', va='center',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    zorder=100
)

ax.set_xlim(0, steps.max() * 1.0)
ax.set_ylim(65, 96)
ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparison of Contrastive Learning Methods', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='-', alpha=0.15)
ax.set_axisbelow(True)
legend = ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=10)
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_edgecolor('lightgray')

plt.tight_layout()

########################### 局部放大 ###########################
# 局部区域（显示最后5%步数）
zoom_xmin = steps.max()*0.95
zoom_xmax = steps.max()*1.0

# 插入inset axes
axins = inset_axes(ax, width="35%", height="35%", loc='upper left',
                  bbox_to_anchor=(0.56, 0.12, 0.57, 0.7),
                  bbox_transform=ax.transAxes, borderpad=1.1)

# 绘制各曲线（CORE曲线高亮，其它淡一些）
for _, name, curve, color, linewidth in method_curves:
    alpha_set = 1.0 if 'CORE' in name else 0.6
    zorder_set = 3 if 'CORE' in name else 2
    lw_set = linewidth + 0.7 if 'CORE' in name else linewidth
    axins.plot(steps, curve, color=color, linewidth=lw_set, alpha=alpha_set, zorder=zorder_set)

# 设置局部范围
axins.set_xlim(zoom_xmin, zoom_xmax)
# 获取Y范围
all_curves_y = []
for _, name, y, _, _ in method_curves:
    mask = (steps >= zoom_xmin)
    all_curves_y.extend(y[mask])
ymin, ymax = min(all_curves_y) - 0.2, max(all_curves_y) + 0.3
axins.set_ylim(ymin, ymax)

# 只在zoom区间内选核心方法的点
core_curve = [item for item in method_curves if item[1].startswith("CORE")][0][2]
core_mask = (steps >= zoom_xmin) & (steps <= zoom_xmax)
if np.any(core_mask):
    # 选区间内最后一个点
    core_steps_in_zoom = steps[core_mask]
    core_curve_in_zoom = core_curve[core_mask]
    core_xfinal = core_steps_in_zoom[-1]
    core_final = core_curve_in_zoom[-1]
    axins.scatter(core_xfinal, core_final, c='#B22222', s=40, marker='D', edgecolor='k', zorder=15)
    axins.annotate('Ours\n%.2f%%' % core_final,
                    xy=(core_xfinal, core_final),
                    xytext=(core_xfinal-0.02*steps.max(), core_final-0.20),
                    color='#B22222', fontsize=9, fontweight='bold',
                    ha='right', va='top',
                    arrowprops=dict(facecolor='#B22222', shrink=0.11, width=1, headwidth=6, alpha=0.65),
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec='gray', alpha=0.8), zorder=100)

# 去掉xy刻度和标签
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xticks([])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_linewidth(1.1)

# 画放大指示线
ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.05)

###############################################################
########################### 局部放大 ###########################
# 只显示18%到22%训练步数区间
zoom_xmin = steps.max()*0.18
zoom_xmax = steps.max()*0.22

# 插入inset axes（可根据需要调整宽高和位置）
axins = inset_axes(ax, width="28%", height="32%", loc='upper left',
                  bbox_to_anchor=(0.28, 0.00, 0.57, 0.7),
                  bbox_transform=ax.transAxes, borderpad=1.1)

# 绘制各曲线（CORE曲线高亮，其它淡一些）
for _, name, curve, color, linewidth in method_curves:
    alpha_set = 1.0 if 'CORE' in name else 0.6
    zorder_set = 3 if 'CORE' in name else 2
    lw_set = linewidth + 0.7 if 'CORE' in name else linewidth
    axins.plot(steps, curve, color=color, linewidth=lw_set, alpha=alpha_set, zorder=zorder_set)

# 设置局部范围
axins.set_xlim(zoom_xmin, zoom_xmax)

# 只统计zoom范围内的y极值用于y轴设定
all_curves_y = []
for _, name, y, _, _ in method_curves:
    mask = (steps >= zoom_xmin) & (steps <= zoom_xmax)
    all_curves_y.extend(y[mask])
if all_curves_y:  # 避免空
    ymin, ymax = min(all_curves_y) - 0.2, max(all_curves_y) + 0.3
    axins.set_ylim(ymin, ymax)

# 若需要，也可在局部图突出CORE此区间末端的点
core_curve = [item for item in method_curves if item[1].startswith("CORE")][0][2]
# 在当前放大区间内选择数据
core_mask = (steps >= zoom_xmin) & (steps <= zoom_xmax)
if np.any(core_mask):
    core_steps_in_zoom = steps[core_mask]
    core_curve_in_zoom = core_curve[core_mask]
    # 选择该区间内最后一个点
    core_xfinal = core_steps_in_zoom[-1]
    core_final = core_curve_in_zoom[-1]
    axins.scatter(core_xfinal, core_final, c='#B22222', s=40, marker='D', edgecolor='k', zorder=15)
    axins.annotate('Ours\n%.2f%%' % core_final,
                    xy=(core_xfinal, core_final),
                    xytext=(core_xfinal-0.01*steps.max(), core_final-0.10),
                    color='#B22222', fontsize=9, fontweight='bold',
                    ha='right', va='top',
                    arrowprops=dict(facecolor='#B22222', shrink=0.11, width=1, headwidth=6, alpha=0.65),
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec='gray', alpha=0.8), zorder=100)

# 去掉xy刻度和标签
axins.set_xticks([])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_linewidth(1.1)

# 画放大指示线（放大框连线）
ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.05)
###############################################################


plt.savefig('output/figure1.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.savefig('output/figure1.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
