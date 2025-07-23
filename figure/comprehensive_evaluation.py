import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, ConnectionPatch
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Set professional IEEE conference style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
    'text.usetex': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'lines.linewidth': 1.5,
    'lines.markersize': 4
})

# Define color scheme for professional appearance
colors = {
    'beacon': '#2E4B99',
    'baseline1': '#D2691E',
    'baseline2': '#8B4513',
    'baseline3': '#556B2F',
    'accent': '#B22222',
    'success': '#228B22',
    'warning': '#FF8C00',
    'neutral': '#708090'
}

def create_comprehensive_evaluation_figure():
    """Generate comprehensive 9-subplot evaluation figure for BEACON paper."""
    
    # Create figure with 3x3 subplot layout
    fig = plt.figure(figsize=(12, 10))
    
    # Subplot (a): Experimental Framework Architecture
    ax_a = plt.subplot(3, 3, 1)
    create_framework_architecture(ax_a)
    ax_a.set_title('(a) Experimental Framework Architecture', fontweight='bold', pad=8)
    
    # Subplot (b): Detection Accuracy Comparison
    ax_b = plt.subplot(3, 3, 2)
    create_accuracy_comparison(ax_b)
    ax_b.set_title('(b) Detection Accuracy Comparison', fontweight='bold', pad=8)
    
    # Subplot (c): Latency Breakdown Analysis
    ax_c = plt.subplot(3, 3, 3)
    create_latency_breakdown(ax_c)
    ax_c.set_title('(c) Latency Breakdown Analysis', fontweight='bold', pad=8)
    
    # Subplot (d): Throughput Scalability Metrics
    ax_d = plt.subplot(3, 3, 4)
    create_throughput_scalability(ax_d)
    ax_d.set_title('(d) Throughput Scalability', fontweight='bold', pad=8)
    
    # Subplot (e): Byzantine Robustness
    ax_e = plt.subplot(3, 3, 5)
    create_byzantine_robustness(ax_e)
    ax_e.set_title('(e) Byzantine Robustness', fontweight='bold', pad=8)
    
    # Subplot (f): Cross-Chain Performance
    ax_f = plt.subplot(3, 3, 6)
    create_crosschain_performance(ax_f)
    ax_f.set_title('(f) Cross-Chain Synchronization', fontweight='bold', pad=8)
    
    # Subplot (g): Edge Deployment Validation
    ax_g = plt.subplot(3, 3, 7)
    create_edge_deployment(ax_g)
    ax_g.set_title('(g) Edge Deployment Validation', fontweight='bold', pad=8)
    
    # Subplot (h): Comparative Analysis
    ax_h = plt.subplot(3, 3, 8)
    create_comparative_analysis(ax_h)
    ax_h.set_title('(h) State-of-the-Art Comparison', fontweight='bold', pad=8)
    
    # Subplot (i): Statistical Significance
    ax_i = plt.subplot(3, 3, 9)
    create_statistical_significance(ax_i)
    ax_i.set_title('(i) Statistical Significance', fontweight='bold', pad=8)
    
    plt.tight_layout(pad=2.0)
    return fig

def create_framework_architecture(ax):
    """Create experimental framework architecture diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # Edge Tier
    edge_box = FancyBboxPatch((0.5, 6), 3, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor=colors['beacon'], linewidth=2)
    ax.add_patch(edge_box)
    ax.text(2, 6.75, 'Edge Tier\n(Jetson AGX)', ha='center', va='center', fontweight='bold')
    
    # Coordination Tier
    coord_box = FancyBboxPatch((4, 4), 4, 2, boxstyle="round,pad=0.1",
                              facecolor='lightgreen', edgecolor=colors['success'], linewidth=2)
    ax.add_patch(coord_box)
    ax.text(6, 5, 'Coordination Tier\n(4×RTX 3090)\nByzantine Consensus', ha='center', va='center', fontweight='bold')
    
    # Blockchain Tier
    blockchain_box = FancyBboxPatch((1, 1), 7, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='lightyellow', edgecolor=colors['warning'], linewidth=2)
    ax.add_patch(blockchain_box)
    ax.text(4.5, 1.75, 'Blockchain Tier (Ethereum, Bitcoin, Binance, Polygon)', ha='center', va='center', fontweight='bold')
    
    # Add arrows
    ax.annotate('', xy=(6, 4), xytext=(2, 6), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.annotate('', xy=(4.5, 2.5), xytext=(6, 4), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

def create_accuracy_comparison(ax):
    """Create detection accuracy comparison across datasets."""
    datasets = ['Ethereum-S', 'Ethereum-P', 'Bitcoin-M', 'Bitcoin-L']
    f1_scores = [97.94, 95.37, 94.26, 98.11]
    std_devs = [0.23, 0.31, 0.28, 0.19]
    
    x_pos = np.arange(len(datasets))
    bars = ax.bar(x_pos, f1_scores, yerr=std_devs, capsize=5, 
                  color=[colors['beacon'], colors['baseline1'], colors['baseline2'], colors['success']],
                  alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_devs[i] + 0.1,
                f'{score:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('F1-Score (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylim(90, 100)
    ax.grid(axis='y', alpha=0.3)

def create_latency_breakdown(ax):
    """Create latency breakdown analysis."""
    components = ['Edge2Seq\nEncode', 'MGD\nLayers', 'Consensus\nAggr.', 'Cross-Chain\nVerif.']
    datasets = ['Ethereum-S', 'Bitcoin-L']
    
    # Latency data (milliseconds)
    eth_s = [8.2, 15.4, 12.6, 5.8]
    btc_l = [10.3, 17.9, 15.8, 7.0]
    
    x = np.arange(len(components))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, eth_s, width, label='Ethereum-S', color=colors['beacon'], alpha=0.8)
    bars2 = ax.bar(x + width/2, btc_l, width, label='Bitcoin-L', color=colors['accent'], alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=6)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=6)
    ax.legend(fontsize=6)
    ax.set_ylim(0, 20)

def create_throughput_scalability(ax):
    """Create throughput scalability metrics."""
    gpu_counts = [1, 2, 4, 8]
    throughput = [2850, 5640, 11280, 20100]  # tx/s
    efficiency = [100, 98.7, 99.4, 88.5]  # percentage
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(gpu_counts, throughput, 'o-', color=colors['beacon'], linewidth=3, 
                    markersize=8, label='Throughput')
    line2 = ax2.plot(gpu_counts, efficiency, 's--', color=colors['accent'], linewidth=2, 
                     markersize=6, label='Efficiency')
    
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Throughput (tx/s)', color=colors['beacon'])
    ax2.set_ylabel('Scaling Efficiency (%)', color=colors['accent'])
    
    # Add grid and styling
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 22000)
    ax2.set_ylim(80, 105)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6)

def create_byzantine_robustness(ax):
    """Create Byzantine robustness analysis."""
    datasets = ['Ethereum-S', 'Ethereum-P', 'Bitcoin-M', 'Bitcoin-L']
    attack_types = ['Model\nPoison', 'Data\nPoison', 'Sybil\nEclipse', 'Delay\nAttack']
    f1_drops = [0.89, 1.20, 1.40, 0.70]
    recovery_times = [2.1, 2.4, 2.6, 1.9]
    
    # Create grouped bar chart
    x = np.arange(len(datasets))
    
    bars1 = ax.bar(x, f1_drops, color=colors['accent'], alpha=0.7, label='F1 Degradation (%)')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x, recovery_times, color=colors['warning'], alpha=0.5, 
                    width=0.6, label='Recovery Time (min)')
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
               f'{f1_drops[i]:.2f}%', ha='center', va='bottom', fontsize=6)
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.05,
                f'{recovery_times[i]:.1f}', ha='center', va='bottom', fontsize=6)
    
    ax.set_ylabel('F1 Degradation (%)', color=colors['accent'])
    ax2.set_ylabel('Recovery Time (min)', color=colors['warning'])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{ds}\n{at}' for ds, at in zip(datasets, attack_types)], fontsize=6)
    ax.set_ylim(0, 2)
    ax2.set_ylim(0, 3)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=6)

def create_crosschain_performance(ax):
    """Create cross-chain synchronization performance."""
    chain_pairs = ['ETH-BTC', 'ETH-BNB', 'BTC-POLY', 'BNB-POLY']
    median_latency = [61, 55, 59, 52]
    p95_latency = [74, 68, 79, 66]
    accuracy = [99.2, 99.7, 99.0, 99.4]
    
    x = np.arange(len(chain_pairs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, median_latency, width, label='Median', 
                   color=colors['beacon'], alpha=0.8)
    bars2 = ax.bar(x + width/2, p95_latency, width, label='P95', 
                   color=colors['accent'], alpha=0.8)
    
    # Add accuracy as text annotations
    for i, acc in enumerate(accuracy):
        ax.text(i, max(median_latency[i], p95_latency[i]) + 2,
               f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=6)
    
    ax.set_ylabel('Sync Latency (ms)')
    ax.set_xlabel('Blockchain Pairs')
    ax.set_xticks(x)
    ax.set_xticklabels(chain_pairs)
    ax.legend(fontsize=6)
    ax.set_ylim(0, 90)
    ax.text(1.5, 85, 'Consensus Accuracy', ha='center', fontweight='bold', fontsize=7)

def create_edge_deployment(ax):
    """Create edge deployment validation."""
    datasets = ['Ethereum-S', 'Ethereum-P', 'Bitcoin-M', 'Bitcoin-L']
    cloud_f1 = [97.94, 95.37, 94.26, 98.11]
    edge_f1 = [94.8, 92.3, 91.7, 93.1]
    energy = [0.94, 1.02, 1.11, 1.19]  # Joules per transaction
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cloud_f1, width, label='Cloud (4×RTX 3090)', 
                   color=colors['beacon'], alpha=0.8)
    bars2 = ax.bar(x + width/2, edge_f1, width, label='Edge (Jetson AGX)', 
                   color=colors['success'], alpha=0.8)
    
    # Add energy consumption as scatter plot
    ax2 = ax.twinx()
    scatter = ax2.scatter(x, energy, c=colors['accent'], s=60, marker='D', 
                         label='Energy (J/tx)', alpha=0.8, edgecolors='black')
    
    ax.set_ylabel('F1-Score (%)')
    ax2.set_ylabel('Energy per Transaction (J)', color=colors['accent'])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylim(85, 100)
    ax2.set_ylim(0.8, 1.3)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=6)

def create_comparative_analysis(ax):
    """Create comparative analysis with state-of-the-art."""
    methods = ['Voronov\n(2024)', 'Song\n(2025)', 'Tharani\n(2024)', 'Chen\n(2025)', 'BEACON\n(Ours)']
    f1_scores = [87.3, 91.2, 89.7, 88.4, 96.2]  # Average BEACON score
    latencies = [340, 180, 220, 280, 46.5]  # Average BEACON latency
    
    # Create bubble chart
    throughputs = [2.1, 4.8, 3.8, 3.2, 10.6]  # k tx/s
    bubble_sizes = [(t/max(throughputs))*300 + 50 for t in throughputs]
    
    colors_list = [colors['baseline1'], colors['baseline2'], colors['baseline3'], 
                  colors['neutral'], colors['beacon']]
    
    scatter = ax.scatter(latencies, f1_scores, s=bubble_sizes, c=colors_list, 
                        alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add method labels
    for i, method in enumerate(methods):
        ax.annotate(method, (latencies[i], f1_scores[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=6, fontweight='bold')
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('F1-Score (%)')
    ax.set_xlim(0, 400)
    ax.set_ylim(85, 98)
    ax.grid(True, alpha=0.3)
    
    # Add legend for bubble sizes
    ax.text(300, 97, 'Bubble size ∝ Throughput', fontsize=6, style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_statistical_significance(ax):
    """Create statistical significance analysis."""
    metrics = ['Accuracy', 'Latency', 'Throughput', 'Byzantine\nTolerance']
    p_values = [0.0001, 0.0003, 0.0002, 0.0001]  # All < 0.001
    effect_sizes = [0.85, 0.92, 0.88, 0.79]  # Cohen's d
    
    # Create bar chart for p-values (log scale)
    log_p_values = [-np.log10(p) for p in p_values]
    bars = ax.bar(metrics, log_p_values, color=[colors['beacon'] if p < 0.001 else colors['warning'] 
                                               for p in p_values], alpha=0.8)
    
    # Add significance threshold line
    ax.axhline(y=-np.log10(0.001), color='red', linestyle='--', linewidth=2, 
              label='p < 0.001')
    
    # Add effect size as text annotations
    for i, (bar, effect) in enumerate(zip(bars, effect_sizes)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'd={effect:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=6)
    
    ax.set_ylabel('-log₁₀(p-value)')
    ax.set_xlabel('Performance Metrics')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=7)
    ax.legend(fontsize=6)
    ax.set_ylim(0, 5)
    ax.text(1.5, 4.5, 'Effect Size (Cohen\'s d)', ha='center', fontweight='bold', fontsize=7)

def save_figure_as_pdf(fig, filename='comprehensive_evaluation.pdf'):
    """Save the figure as a high-quality PDF."""
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300, 
                   facecolor='white', edgecolor='none')
    print(f"Figure saved as {filename}")

# Generate the comprehensive evaluation figure
if __name__ == "__main__":
    # Create the figure
    fig = create_comprehensive_evaluation_figure()
    
    # Save as PDF
    save_figure_as_pdf(fig, 'comprehensive_evaluation.pdf')
    
    # Display the figure
    plt.show()

# Additional function to create a summary statistics table
def print_performance_summary():
    """Print performance summary for reference."""
    print("\n" + "="*60)
    print("BEACON Performance Summary")
    print("="*60)
    
    datasets = {
        'Ethereum-S': {'f1': 97.94, 'latency': 42, 'throughput': 11320},
        'Ethereum-P': {'f1': 95.37, 'latency': 47, 'throughput': 10780},
        'Bitcoin-M': {'f1': 94.26, 'latency': 45, 'throughput': 9930},
        'Bitcoin-L': {'f1': 98.11, 'latency': 51, 'throughput': 10410}
    }
    
    for dataset, metrics in datasets.items():
        print(f"{dataset:12} | F1: {metrics['f1']:5.2f}% | "
              f"Latency: {metrics['latency']:2d}ms | "
              f"Throughput: {metrics['throughput']:5.0f} tx/s")
    
    print("\nByzantine Robustness: <1.4% degradation under 33% malicious nodes")
    print("Cross-chain Synchronization: <80ms P95 latency, 99%+ accuracy")
    print("Edge Deployment: 91.7-94.8% F1-score on Jetson AGX Xavier")
    print("="*60)

# Execute performance summary
print_performance_summary()