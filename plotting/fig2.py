import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Create figure with specific size for wide format (3:1 aspect ratio)
fig, ax = plt.subplots(figsize=(15, 5))
plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)

# Set background color
ax.set_facecolor('white')

# Remove axes
ax.axis('off')

# Add title
fig.suptitle('Comparison of Contrastive Learning Architectures', fontsize=18, fontweight='bold', y=0.95)

# Add vertical divider
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, ymin=0.05, ymax=0.9)

# Add section titles
ax.text(0.25, 0.88, 'Traditional Complex Framework', ha='center', fontsize=16, fontweight='bold')
ax.text(0.75, 0.88, 'CORE (Our Approach)', ha='center', fontsize=16, fontweight='bold')

# Colors
trad_color = '#d0e6ff'
trad_edge = '#4682b4'
core_color = '#d0f0d0'
core_edge = '#3cb371'

# Function to create rectangle
def create_box(x, y, width, height, color, edge_color, label, fontsize=12, sublabel=None, layers=0):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0.02,rounding_size=0.1",
                                 facecolor=color, edgecolor=edge_color, linewidth=2)
    ax.add_patch(rect)
    
    # Main label
    if sublabel:
        ax.text(x + width/2, y + height/2 - 0.02, label, ha='center', va='center', fontsize=fontsize)
        ax.text(x + width/2, y + height/2 + 0.02, sublabel, ha='center', va='center', fontsize=fontsize)
    else:
        ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize)
    
    # Add small rectangles for layers if needed
    if layers > 0:
        layer_height = height * 0.15
        spacing = (height - layers * layer_height) / (layers + 1)
        for i in range(layers):
            y_pos = y + spacing + i * (layer_height + spacing)
            layer_rect = patches.Rectangle((x + width*0.2, y_pos), width*0.6, layer_height,
                                          facecolor=edge_color, alpha=0.3, edgecolor=edge_color, linewidth=1)
            ax.add_patch(layer_rect)

# Function to create arrows
def create_arrow(x_start, y_start, x_end, y_end, color='black', linestyle='-'):
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5, linestyle=linestyle))

# Traditional framework components
create_box(0.05, 0.5, 0.12, 0.15, trad_color, trad_edge, "Backbone\nEncoder")
# 修正这一行，确保sublabel参数是在正确位置
create_box(0.22, 0.5, 0.12, 0.15, trad_color, trad_edge, "Multi-layer", sublabel="Projection Head", layers=3)
create_box(0.22, 0.25, 0.12, 0.15, trad_color, trad_edge, "Predictor\nNetwork")
create_box(0.22, 0.05, 0.12, 0.15, trad_color, trad_edge, "Memory Bank")
create_box(0.39, 0.25, 0.1, 0.3, trad_color, trad_edge, "Complex\nLoss Function")

# Add formula to loss function
ax.text(0.44, 0.4, r"$\Sigma L_i = L_{pos} + L_{neg}$", ha='center', fontsize=10)
ax.text(0.44, 0.35, r"$+ L_{reg} + L_{distill}$", ha='center', fontsize=10)

# Traditional framework arrows
create_arrow(0.17, 0.575, 0.22, 0.575)  # backbone to projection
create_arrow(0.28, 0.5, 0.28, 0.4)  # projection to predictor
create_arrow(0.34, 0.575, 0.39, 0.45)  # projection to loss
create_arrow(0.34, 0.325, 0.39, 0.4)  # predictor to loss
create_arrow(0.28, 0.25, 0.28, 0.2)  # predictor to memory
create_arrow(0.22, 0.125, 0.15, 0.125)  # memory to feedback
create_arrow(0.15, 0.125, 0.15, 0.5)  # feedback to backbone
create_arrow(0.44, 0.25, 0.44, 0.125)  # loss to memory
create_arrow(0.44, 0.125, 0.34, 0.125)  # feedback to memory

# Core approach components
create_box(0.6, 0.35, 0.12, 0.15, core_color, core_edge, "Backbone\nEncoder")
create_box(0.77, 0.35, 0.12, 0.15, core_color, core_edge, "Simple\nProjection Head")
create_box(0.94, 0.35, 0.12, 0.15, core_color, core_edge, "Simple\nContrastive Loss")

# Core approach arrows
create_arrow(0.72, 0.425, 0.77, 0.425)
create_arrow(0.89, 0.425, 0.94, 0.425)

# Add notes highlighting simplicity
ax.text(0.75, 0.15, "Streamlined architecture with minimal components",
        ha='center', fontsize=14, fontstyle='italic')

plt.savefig('output/contrastive_learning_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('output/contrastive_learning_comparison.svg', bbox_inches='tight')
plt.show()
