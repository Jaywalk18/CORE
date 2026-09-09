import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle


raw_data = pd.DataFrame([
    ['simclr', 10, 59.4899, 60.73477585, 1558.465283/1024, 823.254004, 0.038172105, 0.072261902],
    ['core',   10, 80.4199, 62.66651411, 1663.710645/1048, 798.2810534, 0.048337672, 0.100741336],
    ['supcon', 10, 74.29,   62.75443864, 1558.465283/1024, 796.7742684, 0.047668691, 0.093238453],
    ['nnclr',  10, 53.2099, 62.91225584, 1783.433496/1024, 794.7690832, 0.029835651, 0.066950138],
    ['wmse',   10, 40.8899, 63.39794638, 1588.991846/1024, 789.0590658, 0.025733235, 0.051821089],
    ['barlow', 10, 50.9199, 63.73973913, 1844.108838/1024, 784.448333, 0.027612199, 0.064911732],
    ['vicreg', 10, 64.3199, 64.21991153, 1815.164844/1024, 779.0874407, 0.035434743, 0.082557999],
    ['simsiam',10, 40.3699, 64.34896228, 1711.82959/1024, 777.109721, 0.023582896, 0.051948778],
    ['mocov2+',10, 38.36,   72.3272332, 1896.487891/1024, 691.3070167, 0.020226863, 0.055489094],
    ['byol',   10, 40.4199, 76.56639862, 1826.142285/1024, 654.0652661, 0.022134037, 0.061797961],
    ['all4one',10, 49.5999, 78.27549977, 1986.015137/1024, 638.7792387, 0.024974583, 0.077647953]
], columns=[
    'name', 'max_epochs', 'val_acc1', 'epoch_time_sec', 'max_cuda_mem_GB',
    'throughput_imgs_per', 'acc1_per_mem', 'acc1_per_throughput'
])
 
# 补充 early-acc
early_acc = [
    45.04999924, 65.65000153, 55.72999954, 35.09999847, 31.62999916,
    34.95999908, 46.66999817, 30.76999855, 25.65999985, 30.31999969, 39.61999893
]
 
# 规范名称
raw_data['Early-ACC'] = early_acc
 
# 只保留你要的字段
result = raw_data[[
    'name', 'max_epochs', 'val_acc1', 'epoch_time_sec', 'max_cuda_mem_GB',
    'throughput_imgs_per', 'acc1_per_mem', 'acc1_per_throughput', 'Early-ACC'
]]
 

raw_data['max_cuda_mem_GB'] = np.sqrt(raw_data['max_cuda_mem_GB'])
# 只选择主力模型参与比较
model_aliases = {
    'CORE(Ours)': 'core',
    'SupCon': 'supcon',
    'SimCLR': 'simclr',
    'VICReg': 'vicreg',
    'All4One': 'all4one',
    'BYOL': 'byol',

}

# 展示的雷达图指标

metrics = [
    ("Mem", 'max_cuda_mem_GB', 'min'),       # 内存效率 
    ("Acc@1", 'val_acc1', 'max'),          # 准确率
    ("Thpt", 'throughput_imgs_per', 'max'),  # Throughput
    ("Early", 'Early-ACC', 'max'),            # 早期准确率
    ("Time", 'epoch_time_sec', 'min'),       # 速度
    ("A/M", 'acc1_per_mem', 'max'),          # 吞吐量效率
]

# ======================== 数据归一化 ========================
def column_minmax(series, bigger_better=True, min_score=30):
    """
    对一列数据进行归一化，保留最小分数为min_score，支持极大或极小最优。
    """
    arr = series.values
    delta = arr.max() - arr.min() + 1e-8
    frac = (arr - arr.min()) / delta if bigger_better else (arr.max() - arr) / delta
    frac = min_score/100 + frac * (1 - min_score/100)
    return frac * 100

norm_data = []
for model_eng in model_aliases.values():
    row = raw_data[raw_data['name'] == model_eng]
    norm = []
    for (label, colname, mode) in metrics:
        higher_better = (mode == 'max')
        score = float(column_minmax(raw_data[colname], bigger_better=higher_better, min_score=30)[row.index[0]])
        norm.append(score)
    norm_data.append(norm)
norm_data = np.array(norm_data)


# ======================== 绘图参数设置 ========================
# 全局美化
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 色板
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#b07aa1']
N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
max_core_idx = np.argmax(norm_data[0])
rot = np.pi/2 - angles[max_core_idx]  # 旋转使主模型高点朝上
tmp = np.deg2rad(30+240)  # 正30度
# 画布
fig, ax = plt.subplots(figsize=(2.8, 2.8), subplot_kw=dict(polar=True), dpi=450)
ax.set_theta_offset(rot+tmp)
ax.set_theta_direction(-1)
plt.xticks([])  # 去除极坐标默认横坐标

# 刻度与网格 - 修改为更直接的放射状网格线
max_scale = 100
scale_ticks = [20, 40, 60, 80]

# 绘制环形网格
for r in scale_ticks:
    # 画同心圆环
    circle = Circle((0, 0), r, transform=ax.transData._b, fill=False, 
                    edgecolor='#cccccc', linestyle='-', linewidth=0.6, alpha=0.6, zorder=0)
    ax.add_patch(circle)

# 最外圈使用黑色实线
# 最外圈使用纯黑色实线
outer_circle = Circle((0, 0), max_scale, transform=ax.transData._b, fill=False, 
                    edgecolor='black', linestyle='-', linewidth=1, alpha=1.0, zorder=2)
ax.add_patch(outer_circle)

ax.add_patch(outer_circle)

# 添加从中心到各轴的放射状线条
for angle in angles:
    ax.plot([angle, angle], [0, max_scale], color='#cccccc', 
            linestyle='-', linewidth=0.6, alpha=0.6, zorder=0)

ax.set_facecolor('#fcfcfc')
ax.spines['polar'].set_visible(False)
ax.set_rlabel_position(67.5)
plt.yticks(scale_ticks , [str(t) for t in scale_ticks] , color='#666666', fontsize=8)
# plt.yticks(scale_ticks + [max_scale], [str(t) for t in scale_ticks + [max_scale]], color='#666666', fontsize=8)
ax.set_ylim(0, max_scale + 14)

# ======================== 放置角标签 ========================
# 为不同标签设置不同的半径，避免遮挡
tmp = 1.05
label_radii = [
    max_scale+25*tmp,  # Mem
    max_scale+39*tmp,  # ACC1
    max_scale+35*tmp,  # thpt
    max_scale+28*tmp,  # E-Acc
    max_scale+30*tmp,  # Time
    max_scale+35*tmp   # A/M
]

# 为不同标签设置微调的角度
angle_offsets = [0.0, 0.25, -0.0, -0.05, 0.15, -0.0]  # Mem # Acc1 # Thpt # E-Acc # Time # Acc-1

label_box = dict(facecolor='white', alpha=0.95, edgecolor='#dddddd', boxstyle="round,pad=0.3")
fontparams = dict(fontsize=10, fontweight='bold', fontname='Times New Roman', zorder=20)
for i, (angle, (label, _, _)) in enumerate(zip(angles, metrics)):
    # 应用角度微调
    adjusted_angle = angle + angle_offsets[i]
    
    ha, va = 'center', 'center'
    if 0 <= adjusted_angle < np.pi/4 or 7*np.pi/4 <= adjusted_angle <= 2*np.pi:
        ha, va = 'center', 'bottom'
    elif np.pi/4 <= adjusted_angle < 3*np.pi/4:
        ha, va = 'left', 'center'
    elif 3*np.pi/4 <= adjusted_angle < 5*np.pi/4:
        ha, va = 'center', 'top'
    elif 5*np.pi/4 <= adjusted_angle < 7*np.pi/4:
        ha, va = 'right', 'center'
    
    # 使用自定义半径
    ax.text(adjusted_angle, label_radii[i], label, ha=ha, va=va, bbox=label_box, **fontparams)

# ======================== 绘制模型曲线 ========================
for i, (model, color) in enumerate(zip(model_aliases.keys(), colors)):
    values = norm_data[i]
    values_closed = np.append(values, values[0])
    angles_closed = np.append(angles, angles[0])

    # 美化不同模型线条样式
    if i == 0:
        lw, alpha_fill, style, mkr, mkrsize, z_order, alpha_line = 2.8, 0.20, '-', 'o', 7, 10, 1.0
    else:
        lw = 1.5
        alpha_fill, style, mkr, mkrsize, z_order, alpha_line = 0.07, ['--', '-.'][i%2], ['s', '^'][i%2], 5, 5, 0.88

    # 曲线与点
    ax.plot(
        angles_closed, values_closed,
        color=color, linewidth=lw, linestyle=style, label=model,
        marker=mkr, markersize=mkrsize, markerfacecolor=color, markeredgecolor='white',
        markeredgewidth=1.0, zorder=z_order, alpha=alpha_line
    )
    ax.fill(angles_closed, values_closed, color=color, alpha=alpha_fill, zorder=1)
    '''
    # 只为主模型标数值
    if i == 0:
        for j, (angle, value) in enumerate(zip(angles, values)):
            ha, va = 'center', 'center'
            if 0 <= angle < np.pi/4 or 7*np.pi/4 <= angle <= 2*np.pi:
                ha, va = 'center', 'bottom'
            elif np.pi/4 <= angle < 3*np.pi/4:
                ha, va = 'left', 'center'
            elif 3*np.pi/4 <= angle < 5*np.pi/4:
                ha, va = 'center', 'top'
            elif 5*np.pi/4 <= angle < 7*np.pi/4:
                ha, va = 'right', 'center'
            ax.text(angle, value+4, str(int(value)), color=color, fontsize=8,
                    fontweight='bold', ha=ha, va=va, zorder=11)
    '''

# ======================== 图例与布局 ========================
legend = ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=len(model_aliases) // 2,  # 每行一半
    fontsize=8,
    frameon=True, facecolor='white', edgecolor='#dddddd', framealpha=0.9,
    handletextpad=0.4, columnspacing=1.5, handlelength=2.5,
)


plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.14)
plt.savefig('output/core_model_radar_optimized.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05, dpi=450)
plt.savefig('output/core_model_radar_optimized.png', dpi=450, bbox_inches='tight', pad_inches=0.05)
plt.show()
