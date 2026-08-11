import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
base_dir = Path(r"C:\Users\PC\OneDrive\Documents\ArtModelThesis\results\Siamese_Training")
output_dir = base_dir.parent

# Define the 3 siamese model configurations (based on best performing backbones)
siamese_configs = [
    ('EfficientNet-B3', 'Brushstroke', 'siamese_efficientnet_b3_brushstroke.json'),
    ('ResNet50', 'Color', 'siamese_resnet50_color.json'),
    ('EfficientNet-B3', 'Texture', 'siamese_efficientnet_b3_texture.json'),
]

def load_siamese_data(filename):
    """Load Siamese network training data"""
    filepath = base_dir / filename
    with open(filepath, 'r') as f:
        return json.load(f)

def create_metric_learning_figure():
    """Create figure showing metric learning efficacy with proper spacing"""
    fig = plt.figure(figsize=(18, 14))
    
    # Create subplots - use gridspec for better control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35, top=0.88, bottom=0.12)
    
    # Top row: two plots side by side
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    # Bottom row: one plot centered (skip first and last column)
    ax3 = fig.add_subplot(gs[1, 1:3])
    
    axes = [ax1, ax2, ax3]
    
    # Main title with proper positioning
    fig.text(0.5, 0.95, 'Metric Learning Efficacy - Distance Separation', 
             ha='center', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.92, '(Widening Gap indicates Style Learning)', 
             ha='center', fontsize=16, style='italic')
    
    colors = {
        'pos': '#27ae60',  # Green for same artist
        'neg': '#e74c3c',  # Red for different artist
        'gap': '#ecf0f1',  # Light gray for margin/gap
    }
    
    legend_handles = None
    legend_labels = None
    
    for idx, (arch, feature, filename) in enumerate(siamese_configs):
        ax = axes[idx]
        
        try:
            data = load_siamese_data(filename)
            history = data['history']
            
            epochs = [h['epoch'] for h in history]
            pos_dists = [h['avg_pos_dist'] for h in history]
            neg_dists = [h['avg_neg_dist'] for h in history]
            
            # Fill the gap (margin) between positive and negative distances
            fill = ax.fill_between(epochs, pos_dists, neg_dists, 
                           color=colors['gap'], alpha=0.35, label='Margin (The Gap)')
            
            # Plot lines with better styling
            line1 = ax.plot(epochs, pos_dists, 'o-', color=colors['pos'], 
                   linewidth=3.5, markersize=11, label='Avg Positive Distance (Same Artist)',
                   markeredgewidth=2, markeredgecolor='white', alpha=0.95, zorder=3)
            line2 = ax.plot(epochs, neg_dists, 'X-', color=colors['neg'], 
                   linewidth=3.5, markersize=13, label='Avg Negative Distance (Diff Artist)',
                   markeredgewidth=2.5, alpha=0.95, zorder=3)
            
            # Capture legend handles from first plot
            if idx == 0:
                legend_handles = [fill, line1[0], line2[0]]
                legend_labels = ['Margin (The Gap)', 
                               'Avg Positive Distance (Same Artist)', 
                               'Avg Negative Distance (Diff Artist)']
            
            # Calculate statistics
            final_margin = neg_dists[-1] - pos_dists[-1]
            initial_margin = neg_dists[0] - pos_dists[0]
            margin_increase = ((final_margin - initial_margin) / initial_margin) * 100
            
            # Add statistics box with better positioning
            stats_text = (f'Initial Gap: {initial_margin:.2f}\n'
                         f'Final Gap: {final_margin:.2f}\n'
                         f'Increase: +{margin_increase:.1f}%')
            ax.text(0.04, 0.96, stats_text, 
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=12, bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                                        alpha=0.97, edgecolor='#555', linewidth=2),
                   fontweight='bold', zorder=5)
            
            # Styling
            ax.set_xlabel('Epochs', fontsize=15, fontweight='bold', labelpad=10)
            ax.set_ylabel('Euclidean Distance', fontsize=15, fontweight='bold', labelpad=10)
            
            # Title with backbone info
            title_text = f'{feature} ({arch})'
            ax.set_title(title_text, fontsize=16, fontweight='bold', pad=12)
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
            ax.set_facecolor('#f8f9fa')
            
            # Set axis limits
            ax.set_xlim([0.5, max(epochs) + 0.5])
            y_min = min(min(pos_dists), min(neg_dists)) - 0.5
            y_max = max(max(pos_dists), max(neg_dists)) + 0.5
            ax.set_ylim([y_min, y_max])
            
            # Increase tick label size
            ax.tick_params(axis='both', labelsize=12)
            
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, f'Data Not Found\n{arch}\n{feature}', 
                   ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='red', fontweight='bold')
            ax.set_title(f'{feature} ({arch})', fontsize=16, fontweight='bold', pad=12)
            print(f"Warning: File not found - {filename}")
    
    # Add single legend below all plots, centered
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12,
                  framealpha=0.97, edgecolor='#555', fancybox=True)
    
    # Save figure
    output_path = output_dir / 'siamese_metric_learning_efficacy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

def create_convergence_figure():
    """Create figure showing Siamese network convergence with proper spacing"""
    fig = plt.figure(figsize=(18, 14))
    
    # Create subplots - use gridspec for better control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=1.00, top=0.90, bottom=0.12, left=0.08, right=0.92)
    
    # Top row: two plots side by side
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    # Bottom row: one plot centered (skip first and last column)
    ax3 = fig.add_subplot(gs[1, 1:3])
    
    axes = [ax1, ax2, ax3]
    
    # Main title with proper positioning
    fig.text(0.5, 0.95, 'Siamese Network Training Convergence', 
             ha='center', fontsize=20, fontweight='bold')
    
    colors = {
        'loss': '#e74c3c',      # Red
        'acc': '#3498db',       # Blue
    }
    
    legend_handles = None
    legend_labels = None
    
    for idx, (arch, feature, filename) in enumerate(siamese_configs):
        ax = axes[idx]
        
        # Create twin axis for accuracy
        ax2 = ax.twinx()
        
        try:
            data = load_siamese_data(filename)
            history = data['history']
            
            epochs = [h['epoch'] for h in history]
            losses = [h['loss'] for h in history]
            accs = [h['triplet_acc'] * 100 for h in history]  # Convert to percentage
            
            # Plot loss on primary axis
            line1 = ax.plot(epochs, losses, '-', color=colors['loss'], 
                           linewidth=4.5, label='Triplet Loss', alpha=0.9, zorder=2)
            
            # Plot accuracy on secondary axis
            line2 = ax2.plot(epochs, accs, '--', color=colors['acc'], 
                            linewidth=4.5, label='Triplet Accuracy', alpha=0.9, zorder=2)
            
            # Capture legend handles from first plot
            if idx == 0:
                legend_handles = line1 + line2
                legend_labels = ['Triplet Loss', 'Triplet Accuracy']
            
            # Statistics
            best_acc = max(accs)
            final_loss = losses[-1]
            loss_reduction = ((losses[0] - losses[-1]) / losses[0]) * 100
            
            # Add statistics box - positioned top-left, higher zorder to prevent overlap
            stats_text = (f'Best Acc: {best_acc:.2f}%\n'
                         f'Final Loss: {final_loss:.4f}\n'
                         f'Loss ↓: {loss_reduction:.1f}%')
            ax.text(0.04, 0.96, stats_text, 
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=12, bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                                        alpha=0.97, edgecolor='#555', linewidth=2),
                   zorder=10, fontweight='bold')
            
            # Styling with INCREASED label padding to prevent overlap
            ax.set_xlabel('Epochs', fontsize=15, fontweight='bold', labelpad=10)
            ax.set_ylabel('Triplet Loss', fontsize=15, fontweight='bold', 
                         color=colors['loss'], labelpad=15)  # Increased from 10
            ax2.set_ylabel('Triplet Accuracy (%)', fontsize=15, fontweight='bold', 
                          color=colors['acc'], labelpad=15)  # Increased from 10
            
            # Title with backbone info
            title_text = f'{feature} ({arch})'
            ax.set_title(title_text, fontsize=16, fontweight='bold', pad=12)
            
            # Color the y-axis labels and add padding
            ax.tick_params(axis='y', labelcolor=colors['loss'], labelsize=12, pad=8)
            ax2.tick_params(axis='y', labelcolor=colors['acc'], labelsize=12, pad=8)
            ax.tick_params(axis='x', labelsize=12)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, zorder=1)
            ax.set_facecolor('#f8f9fa')
            
            # Set reasonable limits
            ax2.set_ylim([min(accs) - 2, 100.5])
            ax.set_xlim([0.5, max(epochs) + 0.5])
            
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, f'Data Not Found\n{arch}\n{feature}', 
                   ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='red', fontweight='bold')
            ax.set_title(f'{feature} ({arch})', fontsize=16, fontweight='bold', pad=12)
            print(f"Warning: File not found - {filename}")
    
    # Add single legend below all plots, centered
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12,
                  framealpha=0.97, edgecolor='#555', fancybox=True)
    
    # Save figure
    output_path = output_dir / 'siamese_training_convergence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

def create_all_siamese_figures():
    """Create both Siamese visualization figures"""
    print("="*70)
    print("Creating Siamese Network Visualization Figures")
    print("="*70)
    print("\nBased on best-performing backbones:")
    print("  • Brushstroke: EfficientNet-B3")
    print("  • Color: ResNet50")
    print("  • Texture: EfficientNet-B3")
    print()
    
    print("1. Creating Metric Learning Efficacy figure...")
    try:
        create_metric_learning_figure()
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Creating Training Convergence figure...")
    try:
        create_convergence_figure()
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("All Siamese figures created successfully!")
    print(f"Output directory: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    create_all_siamese_figures()