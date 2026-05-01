import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(22, 15))
ax.set_xlim(0, 22)
ax.set_ylim(0, 15)
ax.axis('off')
fig.patch.set_facecolor('white')

# ---- Colors ----
C_INPUT = ('#E8F5E9', '#2E7D32')
C_B1 = ('#E3F2FD', '#1565C0')
C_B2 = ('#FFF3E0', '#E65100')
C_B3 = ('#F3E5F5', '#7B1FA2')
C_B4 = ('#FFEBEE', '#C62828')
C_B5 = ('#E0F7FA', '#00695C')
C_OUT = ('#F1F8E9', '#558B2F')
C_ARR = '#37474F'
C_RED = '#C62828'

def box(x, y, w, h, colors, lw=1.8, alpha=0.92, z=2, rad=0.06):
    fc, ec = colors
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(p)

def arr(x1, y1, x2, y2, c=C_ARR, lw=1.8, cs="arc3,rad=0", ms=14):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                        mutation_scale=ms, lw=lw, color=c, zorder=3,
                        connectionstyle=cs)
    ax.add_patch(a)

def tx(x, y, s, fs=9, ha='center', va='center', w='normal', c='#212121',
       z=5, st='normal'):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, fontweight=w, color=c,
            zorder=z, fontstyle=st)

def star(x, y, s=80):
    ax.scatter(x, y, marker='*', s=s, c=C_RED, zorder=6, edgecolors='none')

# ======================== TITLE ========================
tx(11, 14.55, 'MutaKernel: A Mutation-Analysis-Driven Testing & Repair Framework',
   fs=17, w='bold', c='#0D47A1')
tx(11, 14.1, 'for LLM-Generated GPU Kernels', fs=13, w='bold', c='#1565C0')

# ======================== ROW 1: Input + Block1 + Block3 ========================

# -- Input --
box(0.3, 10.7, 2.8, 2.9, C_INPUT, lw=2)
tx(1.7, 13.3, 'Input', fs=12, w='bold', c=C_INPUT[1])
tx(1.7, 12.75, 'LLM-Generated', fs=9.5, c='#333')
tx(1.7, 12.35, 'GPU Kernels', fs=9.5, c='#333')
tx(1.7, 11.9, '(Triton / CUDA)', fs=8.5, c='#666', st='italic')
box(0.55, 10.85, 2.3, 0.7, ('white', '#66BB6A'), lw=1.2)
tx(1.7, 11.3, 'KernelBench', fs=9, w='bold', c='#2E7D32')
tx(1.7, 11.0, 'Level 1 / 2 / 3', fs=8, c='#555')

# -- Block 1: MutOperators --
box(3.8, 10.7, 6.8, 2.9, C_B1, lw=2.2)
tx(7.2, 13.3, 'Block 1: MutOperators', fs=12, w='bold', c=C_B1[1])
tx(7.2, 12.85, 'ML-Aware Mutation Operator Suite (16 Operators)', fs=9, c='#444')

cw, ch = 3.1, 0.45
box(4.05, 12.0, cw, ch, ('#BBDEFB', C_B1[1]), lw=1, alpha=0.7)
tx(5.6, 12.225, 'A: Arithmetic (3)', fs=8.5, c='#1565C0')
box(7.35, 12.0, cw, ch, ('#BBDEFB', C_B1[1]), lw=1, alpha=0.7)
tx(8.9, 12.225, 'B: GPU Parallel (4)', fs=8.5, c='#1565C0')

box(4.05, 11.3, cw, ch, ('#FFCDD2', C_RED), lw=1.6, alpha=0.85)
tx(5.6, 11.525, 'C: ML Numerical (7)', fs=8.5, w='bold', c=C_RED)
star(4.18, 11.525, s=70)
box(7.35, 11.3, cw, ch, ('#FFCDD2', C_RED), lw=1.6, alpha=0.85)
tx(8.9, 11.525, 'D: LLM Patterns (2)', fs=8.5, w='bold', c=C_RED)
star(7.48, 11.525, s=70)

tx(7.2, 10.92, 'StabRemove | AccDowngrade | EpsilonModify | ScaleModify | CastRemove | ...',
   fs=7, c='#888', st='italic')

# -- Block 3: RealismGuard --
box(11.4, 10.7, 3.2, 2.9, C_B3, lw=2.2)
tx(13.0, 13.3, 'Block 3:', fs=11, w='bold', c=C_B3[1])
tx(13.0, 12.85, 'RealismGuard', fs=11, w='bold', c=C_B3[1])
tx(13.0, 12.3, 'Mutant Realism', fs=9, c='#444')
tx(13.0, 11.95, 'Validation', fs=9, c='#444')
box(11.65, 10.85, 2.7, 0.8, ('#EDE7F6', '#9C27B0'), lw=1.1, alpha=0.8)
tx(13.0, 11.35, 'Pattern Matching', fs=8, c='#6A1B9A')
tx(13.0, 11.0, 'on Real LLM Bugs (102)', fs=7.5, c='#6A1B9A')

# ======================== ROW 2: Block 2 + Results ========================

# -- Block 2: MutEngine --
box(0.3, 5.9, 10.3, 4.3, C_B2, lw=2.2)
tx(5.45, 9.9, 'Block 2: MutEngine', fs=12, w='bold', c=C_B2[1])
tx(5.45, 9.5, 'GPU Kernel Mutation Testing Execution Engine', fs=9, c='#444')

# Pipeline
stages = [
    (0.7, 8.3, 2.5, 0.8, 'Parse\n(Triton / CUDA)'),
    (3.6, 8.3, 2.5, 0.8, 'Mutate\n(16 Operators)'),
    (6.5, 8.3, 3.8, 0.8, 'Compile & Execute\n(GPU JIT)'),
]
for sx, sy, sw, sh, sl in stages:
    box(sx, sy, sw, sh, ('#FFE0B2', C_B2[1]), lw=1.2, alpha=0.8)
    tx(sx + sw/2, sy + sh/2, sl, fs=8, c='#BF360C')
arr(3.2, 8.7, 3.6, 8.7, C_B2[1], lw=1.5)
arr(6.1, 8.7, 6.5, 8.7, C_B2[1], lw=1.5)

# Comparison & Equiv
box(0.7, 6.3, 4.5, 1.4, ('#FFF8E1', '#FFA000'), lw=1.3, alpha=0.8)
tx(2.95, 7.35, 'Output Comparison', fs=9, w='bold', c='#E65100')
tx(2.95, 6.95, 'torch.allclose (Multi-seed, atol/rtol)', fs=7.5, c='#795548')
tx(2.95, 6.6, '5 Random Seeds x Configurable Tolerances', fs=7, c='#8D6E63')

box(5.6, 6.3, 4.7, 1.4, ('#FFF8E1', '#FFA000'), lw=1.3, alpha=0.8)
tx(7.95, 7.35, 'Equivalent Mutant Detector', fs=9, w='bold', c='#E65100')
tx(7.95, 6.95, 'Syntax Normalization + 100x Bitwise Check', fs=7.5, c='#795548')
tx(7.95, 6.6, 'torch.equal (strict) for equivalence', fs=7, c='#8D6E63')

arr(5.2, 7.0, 5.6, 7.0, C_B2[1], lw=1.3)

tx(5.45, 6.08, 'Killed  |  Survived  |  Stillborn  |  Equivalent', fs=8.5, c='#666', st='italic')

# -- Mutation Results --
box(11.4, 7.5, 3.2, 2.4, ('#ECEFF1', '#546E7A'), lw=1.8)
tx(13.0, 9.6, 'Mutation Results', fs=10.5, w='bold', c='#37474F')
tx(13.0, 9.1, 'Mutation Score', fs=9, c='#555')
tx(13.0, 8.7, '(per Category / Operator)', fs=8, c='#777')
tx(13.0, 8.25, 'JSON + Markdown Report', fs=7.5, c='#888', st='italic')
tx(13.0, 7.8, 'RQ1: Detection Adequacy', fs=7.5, c='#9E9E9E', st='italic')

# -- Survived Mutants --
box(11.4, 5.9, 3.2, 1.3, ('#EFEBE9', '#795548'), lw=1.8)
tx(13.0, 6.95, 'Survived Mutants', fs=10, w='bold', c='#4E342E')
tx(13.0, 6.5, '+ Mutation Site Info', fs=8, c='#6D4C41')
tx(13.0, 6.12, '+ Operator Category', fs=8, c='#6D4C41')

# ======================== ROW 3: Block 4 ========================

box(7.5, 0.7, 14.2, 4.8, C_B4, lw=2.2)
tx(14.6, 5.2, 'Block 4: MutRepair  --  Mutation-Analysis-Guided Feedback Repair',
   fs=12, w='bold', c=C_B4[1])

# Enhanced Input
box(7.8, 3.8, 4.0, 1.05, ('#FFCDD2', '#E53935'), lw=1.3, alpha=0.8)
tx(9.8, 4.45, 'Enhanced Input Generator', fs=9, w='bold', c='#B71C1C')
tx(9.8, 4.05, '11 Strategies | Shape-Preserving', fs=7.5, c='#C62828')

# Feedback Builder
box(12.2, 3.8, 4.0, 1.05, ('#FFCDD2', '#E53935'), lw=1.3, alpha=0.8)
tx(14.2, 4.45, 'Feedback Builder', fs=9, w='bold', c='#B71C1C')
tx(14.2, 4.05, 'B0 / B1 / B2 / B3 / Ours (5 modes)', fs=7.5, c='#C62828')
arr(11.8, 4.3, 12.2, 4.3, C_B4[1], lw=1.5)

# Repair Loop
box(7.8, 2.2, 4.0, 1.2, ('#FFEBEE', '#EF5350'), lw=1.3, alpha=0.8)
tx(9.8, 3.0, 'Repair Loop (max N rounds)', fs=9, w='bold', c='#B71C1C')
tx(9.8, 2.6, 'LLM Call -> Code Extract -> Dual Verify', fs=7.5, c='#C62828')
tx(9.8, 2.32, '(Standard Test + Enhanced Test)', fs=7, c='#C62828')
arr(12.5, 3.8, 10.5, 3.4, C_B4[1], lw=1.3, cs="arc3,rad=0.15")

# Experience Store
box(12.2, 2.2, 4.0, 1.2, ('#FFEBEE', '#EF5350'), lw=1.3, alpha=0.8)
tx(14.2, 3.0, 'Experience Store', fs=9, w='bold', c='#B71C1C')
tx(14.2, 2.6, 'Record Successful Repair Diffs', fs=7.5, c='#C62828')
tx(14.2, 2.32, 'JSONL -> Feed to MutEvolve', fs=7, c='#C62828')
arr(11.8, 2.8, 12.2, 2.8, C_B4[1], lw=1.5)

# Causal Isolation
box(16.7, 2.6, 4.7, 2.3, ('#FBE9E7', '#FF5722'), lw=1.3, alpha=0.75)
tx(19.05, 4.55, 'Causal Isolation Design', fs=9, w='bold', c='#BF360C')
tx(19.05, 4.1, 'B0: error only', fs=7.5, c='#5D4037', ha='center')
tx(19.05, 3.75, 'B1: + general hint', fs=7.5, c='#5D4037')
tx(19.05, 3.4, 'B2: + failing inputs', fs=7.5, c='#5D4037')
tx(19.05, 3.05, 'B3: + code location', fs=7.5, c='#5D4037')
tx(19.05, 2.7, 'Ours: + mutation analysis', fs=7.5, w='bold', c='#BF360C')

# Output
box(16.7, 0.85, 4.7, 1.4, C_OUT, lw=2)
tx(19.05, 1.95, 'Output: Repaired Kernels', fs=10, w='bold', c=C_OUT[1])
tx(19.05, 1.5, '+ Per-Mode Repair Rate (RQ2 / RQ3)', fs=8, c='#555')
tx(19.05, 1.1, '+ Statistical Significance Tests', fs=7.5, c='#888', st='italic')

arr(19.05, 2.6, 19.05, 2.25, C_ARR, lw=2)

# ======================== Block 5: MutEvolve (bottom left) ========================

box(0.3, 0.7, 6.8, 4.8, C_B5, lw=2.2)
tx(3.7, 5.2, 'Block 5: MutEvolve', fs=12, w='bold', c=C_B5[1])
tx(3.7, 4.75, 'Adaptive Mutation Rule Evolution (Ablation: RQ4)', fs=9, c='#444')

box(0.6, 3.4, 2.9, 1.0, ('#B2EBF2', '#00838F'), lw=1.2, alpha=0.8)
tx(2.05, 4.1, 'Pattern Miner', fs=9, w='bold', c='#004D40')
tx(2.05, 3.7, 'Normalize | Filter | Merge', fs=7.5, c='#00695C')

box(3.8, 3.4, 2.9, 1.0, ('#B2EBF2', '#00838F'), lw=1.2, alpha=0.8)
tx(5.25, 4.1, 'Rule Generator', fs=9, w='bold', c='#004D40')
tx(5.25, 3.7, 'DynamicOperator (Cat. E)', fs=7.5, c='#00695C')
arr(3.5, 3.9, 3.8, 3.9, C_B5[1], lw=1.5)

tx(3.7, 2.7, 'Repair Experiences -> Mine Frequent Patterns', fs=8, c='#555', st='italic')
tx(3.7, 2.3, '-> Generate New Mutation Operators', fs=8, c='#555', st='italic')
tx(3.7, 1.7, 'Ablation: w/ vs w/o E-operators', fs=8, c='#999', st='italic')
tx(3.7, 1.3, 'on second-half problems', fs=8, c='#999', st='italic')

# ======================== CONNECTING ARROWS ========================

# Input -> Block 1
arr(3.1, 12.15, 3.8, 12.15, C_ARR, lw=2.5)

# Block 1 -> Block 2 (operators)
arr(7.2, 10.7, 5.45, 10.2, C_B1[1], lw=2, cs="arc3,rad=0")
tx(7.5, 10.4, 'Operators', fs=8, c=C_B1[1], st='italic')

# Block 1 -> Block 3
arr(10.6, 12.15, 11.4, 12.15, C_ARR, lw=2)

# Block 2 -> Mutation Results
arr(10.6, 8.5, 11.4, 8.7, C_ARR, lw=2)

# Block 2 -> Survived
arr(10.6, 6.5, 11.4, 6.5, C_ARR, lw=2)

# Survived -> Block 4
arr(13.0, 5.9, 13.0, 5.5, C_ARR, lw=2.2)
tx(14.5, 5.7, 'Survived + Site Info', fs=8, c='#795548', st='italic')

# Block 4 Experience -> Block 5
arr(12.2, 2.5, 7.1, 3.0, C_B5[1], lw=2, cs="arc3,rad=-0.15")
tx(9.5, 2.05, 'Repair Experiences', fs=8, c=C_B5[1], st='italic')

# Block 5 -> Block 1 (feedback loop - evolved operators)
arr(3.7, 5.5, 5.5, 10.7, '#00695C', lw=2.2, cs="arc3,rad=-0.5")
tx(1.0, 8.3, 'New Operators', fs=9, c='#00695C', w='bold')
tx(1.0, 7.85, '(Category E)', fs=8.5, c='#00695C', w='bold')

# ======================== DASHED CONTRIBUTION FRAMES ========================

r1 = Rectangle((0.1, 5.6), 14.8, 8.2, lw=1.8, ec=C_B1[1],
               fc='none', ls='--', zorder=1, alpha=0.45)
ax.add_patch(r1)
tx(7.5, 13.75, 'Contribution 1: MutEngine + ML-Aware Operators (Section 3)',
   fs=9, w='bold', c=C_B1[1], st='italic')

r2 = Rectangle((7.3, 0.5), 14.5, 5.2, lw=1.8, ec=C_B4[1],
               fc='none', ls='--', zorder=1, alpha=0.45)
ax.add_patch(r2)
tx(17.5, 5.55, 'Contribution 2: MutRepair (Section 4)', fs=9, w='bold', c=C_B4[1], st='italic')

# ======================== LEGEND ========================

ly = 0.15
items = [
    (0.3, ly, C_B1, 'Block 1: MutOperators (Sec 3.2)'),
    (4.0, ly, C_B2, 'Block 2: MutEngine (Sec 3.1, 3.4)'),
    (8.0, ly, C_B3, 'Block 3: RealismGuard (Sec 3.3)'),
    (12.0, ly, C_B4, 'Block 4: MutRepair (Sec 4)'),
    (16.0, ly, C_B5, 'Block 5: MutEvolve (Sec 5)'),
]
for lx, lny, colors, label in items:
    box(lx, lny, 0.3, 0.3, colors, lw=1.2, alpha=0.9)
    tx(lx + 0.5, lny + 0.15, label, fs=8, ha='left', c='#333')

star(20.0, ly + 0.15, s=80)
tx(20.2, ly + 0.15, '= Novel', fs=8, ha='left', c=C_RED, w='bold')

plt.tight_layout(pad=0.2)
out_base = 'd:/doctor_learning/Academic_Project/paper_1/MutaKernel/figures/method_overview'
plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{out_base}.pdf', bbox_inches='tight', facecolor='white')
print(f"Done! Saved to {out_base}.png and .pdf")
