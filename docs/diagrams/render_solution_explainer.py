#!/usr/bin/env python3
"""Render a slide-ready solution storyboard as SVG and PNG.

Saves:
 - docs/diagrams/solution_explainer_slide.svg
 - docs/diagrams/solution_explainer_slide.png

Designed for 16:9 (1920x1080) output.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch


OUT_SVG = Path(__file__).with_name("solution_explainer_slide.svg")
OUT_PNG = Path(__file__).with_name("solution_explainer_slide.png")


def box(ax, xy, w, h, label_lines, facecolor, edgecolor, fontsize=16, pad=6):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.08,rounding_size=8",
                         linewidth=2, ec=edgecolor, fc=facecolor)
    ax.add_patch(box)
    # center text
    tx = x + w / 2
    ty = y + h / 2
    ax.text(tx, ty, "\n".join(label_lines), ha="center", va="center", fontsize=fontsize, wrap=True)
    return box


def arrow(ax, start, end, lw=2, color="#333333"):
    style = ArrowStyle("Simple", head_length=8, head_width=6, tail_width=0.6)
    a = FancyArrowPatch(start, end, arrowstyle=style, linewidth=lw, color=color, mutation_scale=12)
    ax.add_patch(a)


def render(out_svg: Path, out_png: Path):
    # Figure size: 1920x1080 at 100 dpi -> figsize 19.2 x 10.8
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#FAFAFA')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Colors (from spec)
    col_input = '#EAF2FF'
    col_input_edge = '#3B6FB6'
    col_shared = '#EAF7EF'
    col_shared_edge = '#2E7D4F'
    col_classical = '#FFF3E6'
    col_classical_edge = '#B86B22'
    col_quantum = '#EEF0FF'
    col_quantum_edge = '#4A55C8'
    col_eval = '#F5EAFE'
    col_eval_edge = '#8A3FC2'
    col_output = '#FFF8D9'
    col_output_edge = '#A07800'

    # Title
    ax.text(5, 95, 'Classical vs Quantum Scenario Generation for Financial Contagion Risk', fontsize=28, weight='bold', va='center')
    ax.text(5, 90.5, 'Research question: can quantum-structured scenario generation surface tail contagion scenarios missed by a classical baseline?', fontsize=14, va='center')

    # Top-left: Input Data
    box(ax, (10, 78), 30, 9, ["[INPUT DATA]", "ECB / financial data", "exposures, balances,", "correlations"], facecolor=col_input, edgecolor=col_input_edge, fontsize=13)

    # Shared SystemSpec centered under input
    box(ax, (30, 64), 40, 10, ["[SHARED] SystemSpec", "same nodes, exposures,", "thresholds, marginals", "and correlations"], facecolor=col_shared, edgecolor=col_shared_edge, fontsize=13)

    # Parallel lanes: Classical (left) and Quantum (right)
    left_x = 8
    right_x = 60
    lane_w = 34
    lane_top = 44

    # Classical lane (left)
    ax.text(left_x + lane_w/2, lane_top + 7.5, 'CLASSICAL / NON-QUANTUM ENGINE', ha='center', fontsize=13, weight='semibold')
    box(ax, (left_x, lane_top), lane_w, 7.5, ["Gaussian Copula", "correlated sampling"], facecolor=col_classical, edgecolor=col_classical_edge, fontsize=12)
    box(ax, (left_x, lane_top - 10), lane_w, 7.5, ["Classical Stress Scenarios"], facecolor=col_classical, edgecolor=col_classical_edge, fontsize=12)

    # Quantum lane (right) - emphasized
    ax.text(right_x + lane_w/2, lane_top + 7.5, 'QUANTUM / QUANTUM-INSPIRED ENGINE', ha='center', fontsize=13, weight='semibold', color=col_quantum_edge)
    box(ax, (right_x, lane_top), lane_w, 7.5, ["Entanglement Layout", "clustering → qubits"], facecolor=col_quantum, edgecolor=col_quantum_edge, fontsize=12)
    box(ax, (right_x, lane_top - 10), lane_w, 7.5, ["Quantum Scenario Generator", "PQC / Born-machine"], facecolor=col_quantum, edgecolor=col_quantum_edge, fontsize=12)
    box(ax, (right_x, lane_top - 20), lane_w, 7.5, ["Quantum Stress Scenarios"], facecolor=col_quantum, edgecolor=col_quantum_edge, fontsize=12)

    # Arrows from SystemSpec down to each lane (vertical-ish)
    arrow(ax, (50, 64), (left_x + lane_w/2, lane_top + 7.5))
    arrow(ax, (50, 64), (right_x + lane_w/2, lane_top + 7.5))

    # Convergence box centered beneath lanes
    conv_x = 36
    conv_w = 28
    conv_y = 24
    box(ax, (conv_x, conv_y + 12), conv_w, 6, ["[SHARED]", "Same Scenario Format"], facecolor=col_shared, edgecolor=col_shared_edge, fontsize=12)

    # Arrows from lane outputs down to convergence (straight vertical segments)
    arrow(ax, (left_x + lane_w/2, lane_top - 2.5), (conv_x + conv_w/2, conv_y + 18))
    arrow(ax, (right_x + lane_w/2, lane_top - 12.5), (conv_x + conv_w/2, conv_y + 18))

    # Simulator and evaluation pipeline below convergence
    sim_y = conv_y - 6
    box(ax, (conv_x, sim_y), conv_w, 7, ["[SHARED]", "Deterministic Contagion Simulator", "generator-agnostic cascade model"], facecolor=col_shared, edgecolor=col_shared_edge, fontsize=12)
    box(ax, (conv_x, sim_y - 12), conv_w, 8.5, ["[EVALUATION]", "Metrics + Harness", "collapse frequency, tail loss, cascade size"], facecolor=col_eval, edgecolor=col_eval_edge, fontsize=12)
    box(ax, (conv_x, sim_y - 25), conv_w, 7, ["Final Result", "fair classical-vs-quantum", "systemic-risk comparison"], facecolor=col_output, edgecolor=col_output_edge, fontsize=12)

    # Vertical arrows through shared pipeline
    arrow(ax, (conv_x + conv_w/2, conv_y + 12), (conv_x + conv_w/2, sim_y + 7))
    arrow(ax, (conv_x + conv_w/2, sim_y), (conv_x + conv_w/2, sim_y - 6))
    arrow(ax, (conv_x + conv_w/2, sim_y - 4), (conv_x + conv_w/2, sim_y - 12))

    # Experimental control callout lower-left
    box(ax, (4, 4), 40, 9, ["Experimental Control", "Only the scenario generator changes.", "SystemSpec, simulator, metrics and evaluation stay fixed."], facecolor=(0.98, 0.98, 0.98), edgecolor='#666666', fontsize=11)

    # Legend at bottom right
    lx = 64
    ly = 4
    box(ax, (lx, ly + 8), 14, 4.5, ["Input/Data"], facecolor=col_input, edgecolor=col_input_edge, fontsize=10)
    box(ax, (lx + 15, ly + 8), 18, 4.5, ["Shared / non-quantum"], facecolor=col_shared, edgecolor=col_shared_edge, fontsize=10)
    box(ax, (lx + 34, ly + 8), 10, 4.5, ["Classical"], facecolor=col_classical, edgecolor=col_classical_edge, fontsize=10)
    box(ax, (lx + 45, ly + 8), 10, 4.5, ["Quantum"], facecolor=col_quantum, edgecolor=col_quantum_edge, fontsize=10)
    box(ax, (lx + 56, ly + 8), 12, 4.5, ["Evaluation"], facecolor=col_eval, edgecolor=col_eval_edge, fontsize=10)

    # Footnote notes
    ax.text(4, 1.5, 'Quantum note: quantum involvement happens here only — clustering/entanglement layout, quantum generator, quantum sampling.  •  Classical note: baseline statistical dependence model', fontsize=9)

    # Save
    fig.savefig(out_svg, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    try:
        render(OUT_SVG, OUT_PNG)
        print('Wrote', OUT_SVG, OUT_PNG)
    except Exception as e:
        print('Rendering failed:', e)
        raise
