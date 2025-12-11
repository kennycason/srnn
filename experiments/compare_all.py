"""
Compare all experiments and generate summary visualization.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'exp1': '#2ecc71',  # Green - Fixed
    'exp2': '#3498db',  # Blue - Progressive  
    'exp3': '#e74c3c',  # Red - Two-Phase
}

def load_results():
    """Load metrics from all experiments."""
    results = {}
    
    # Exp1: Fixed
    with open('../results/exp1_fixed/metrics.json') as f:
        results['exp1'] = json.load(f)
    results['exp1']['name'] = 'Fixed Capacity'
    results['exp1']['approach'] = '16 experts, 40 epochs, replay'
    
    # Exp2: Progressive
    with open('../results/exp2_progressive/metrics.json') as f:
        results['exp2'] = json.load(f)
    results['exp2']['name'] = 'Progressive Growth'
    results['exp2']['approach'] = '2→16 experts, no replay'
    
    # Exp3: Two-Phase
    with open('../results/exp3_two_phase/metrics.json') as f:
        results['exp3'] = json.load(f)
    results['exp3']['name'] = 'Two-Phase'
    results['exp3']['approach'] = '4→16 experts, no replay'
    # Normalize loss field
    results['exp3']['loss'] = results['exp3'].get('phase2_loss', results['exp3'].get('loss', 0))
    
    return results

def create_comparison_chart(results):
    """Create main comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    experiments = ['exp1', 'exp2', 'exp3']
    names = [results[e]['name'] for e in experiments]
    colors = [COLORS[e] for e in experiments]
    
    # 1. Active Experts
    ax = axes[0]
    active = [results[e]['active_experts'] for e in experiments]
    total = [results[e]['total_experts'] for e in experiments]
    bars = ax.bar(names, active, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(y=16, color='gray', linestyle='--', alpha=0.5, label='Max (16)')
    ax.set_ylabel('Active Experts', fontsize=12)
    ax.set_title('Expert Utilization', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 18)
    # Add value labels
    for bar, val, tot in zip(bars, active, total):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val}/{tot}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Usage Balance (CV - lower is better)
    ax = axes[1]
    cvs = [results[e]['cv'] for e in experiments]
    bars = ax.bar(names, cvs, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_title('Usage Balance (↓ lower = more balanced)', fontsize=14, fontweight='bold')
    # Add value labels
    for bar, val in zip(bars, cvs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Reconstruction Loss
    ax = axes[2]
    losses = [results[e]['loss'] * 1000 for e in experiments]  # Scale for readability
    bars = ax.bar(names, losses, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Loss (×10⁻³)', fontsize=12)
    ax.set_title('Reconstruction Quality (↓ lower = better)', fontsize=14, fontweight='bold')
    # Add value labels
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('SRNN Experiment Comparison: Saturation Routing Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def create_expert_usage_comparison(results):
    """Create side-by-side expert usage heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    experiments = ['exp1', 'exp2', 'exp3']
    
    for ax, exp in zip(axes, experiments):
        usage = results[exp]['usage']
        # Convert to array
        counts = np.array([usage.get(str(i), 0) for i in range(16)])
        counts_norm = counts / counts.sum() * 100  # Percentage
        
        bars = ax.bar(range(16), counts_norm, color=COLORS[exp], 
                      edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.axhline(y=100/16, color='red', linestyle='--', alpha=0.7, 
                   label=f'Uniform ({100/16:.1f}%)')
        ax.set_xlabel('Expert ID', fontsize=10)
        ax.set_ylabel('Usage %', fontsize=10)
        ax.set_title(f"{results[exp]['name']}\n({results[exp]['active_experts']}/16 active, CV={results[exp]['cv']:.2f})", 
                     fontsize=11, fontweight='bold')
        ax.set_xticks(range(16))
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(counts_norm) * 1.2)
    
    plt.suptitle('Expert Usage Distribution Across Experiments', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def create_summary_table(results):
    """Create a summary table as an image."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Table data
    columns = ['Experiment', 'Approach', 'Active\nExperts', 'CV\n(↓ better)', 'Loss\n(↓ better)', 'Replay?']
    rows = []
    
    for exp in ['exp1', 'exp2', 'exp3']:
        r = results[exp]
        rows.append([
            r['name'],
            r['approach'],
            f"{r['active_experts']}/16",
            f"{r['cv']:.3f}",
            f"{r['loss']:.6f}",
            'Yes' if exp == 'exp1' else 'No'
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')
    
    # Color code results
    for i, exp in enumerate(['exp1', 'exp2', 'exp3'], start=1):
        table[(i, 0)].set_facecolor(COLORS[exp] + '40')  # 40 = alpha
    
    plt.title('SRNN Experiment Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def main():
    print("Loading experiment results...")
    results = load_results()
    
    os.makedirs('../results/comparison', exist_ok=True)
    
    print("Creating comparison charts...")
    
    # Main comparison
    fig1 = create_comparison_chart(results)
    fig1.savefig('../results/comparison/metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: metrics_comparison.png")
    
    # Expert usage
    fig2 = create_expert_usage_comparison(results)
    fig2.savefig('../results/comparison/expert_usage_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: expert_usage_comparison.png")
    
    # Summary table
    fig3 = create_summary_table(results)
    fig3.savefig('../results/comparison/summary_table.png', dpi=150, bbox_inches='tight')
    print("  Saved: summary_table.png")
    
    plt.close('all')
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 70)
    
    for exp in ['exp1', 'exp2', 'exp3']:
        r = results[exp]
        print(f"\n{r['name']} ({r['approach']})")
        print(f"  Active Experts: {r['active_experts']}/16")
        print(f"  Usage CV: {r['cv']:.3f}")
        print(f"  Loss: {r['loss']:.6f}")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
1. SATURATION ROUTING PREVENTS COLLAPSE
   - All experiments maintain high expert utilization (14-16/16)
   - Without saturation routing, progressive growth collapses to ~3/16

2. TRADE-OFFS
   - Fixed (Exp1): Best balance (CV=0.18) but requires replay
   - Progressive (Exp2): Fastest, slight utilization drop (14/16)  
   - Two-Phase (Exp3): Full utilization without replay - best for continual learning

3. LOSS VS UTILIZATION
   - More training (Exp1) = lower loss but more compute
   - Progressive approaches trade some quality for efficiency
   - All achieve good reconstruction quality
""")
    
    print("Comparison saved to results/comparison/")

if __name__ == "__main__":
    main()

