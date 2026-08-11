import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
base_dir = Path(r"C:\Users\PC\OneDrive\Documents\ArtModelThesis\results")
output_dir = base_dir

# Define architectures and features
architectures = ['AlexNet', 'ResNet50', 'VGG16', 'EfficientNet-B3']
features = ['Color', 'Texture', 'Brushstroke']
model_types = ['Global', 'Local']

def load_data(model_type, arch, feature):
    """Load JSON data for a specific configuration"""
    subdir = f"{model_type}_Architecture_Comparison"
    arch_name = arch.lower().replace('-', '_')
    filename = f"{model_type.lower()}_{arch_name}_{feature.lower()}.json"
    filepath = base_dir / subdir / filename
    
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_fold_metrics(data):
    """Extract train and val accuracy for each fold (using best val_acc epoch)"""
    folds = []
    train_accs = []
    val_accs = []
    
    for i in range(1, 6):
        fold_key = f'fold_{i}'
        if fold_key in data:
            folds.append(i)
            val_accs.append(data[fold_key]['best_val_acc'] * 100)
            
            # Get train acc from the epoch with best val_acc
            history = data[fold_key]['history']
            best_val_acc = data[fold_key]['best_val_acc']
            best_epoch = None
            for epoch in history:
                if abs(epoch.get('val_acc', 0) - best_val_acc) < 0.0001:
                    best_epoch = epoch
                    break
            
            if best_epoch:
                train_accs.append(best_epoch.get('train_acc', 0) * 100)
            else:
                # Fallback to last epoch if best not found
                train_accs.append(history[-1].get('train_acc', 0) * 100 if history else 0)
    
    return folds, train_accs, val_accs

def create_architecture_figure(arch_name):
    """Create a figure with 6 subplots (3 features x 2 model types) for one architecture"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'{arch_name} Architecture - Cross-Validation Performance', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Professional color scheme
    colors = {
        'train': '#1f77b4',  # Blue
        'val': '#ff7f0e',    # Orange
    }
    
    for row, model_type in enumerate(model_types):
        for col, feature in enumerate(features):
            ax = axes[row, col]
            
            # Load data
            try:
                data = load_data(model_type, arch_name, feature)
                folds, train_accs, val_accs = extract_fold_metrics(data)
                
                # Calculate statistics
                mean_train = np.mean(train_accs)
                mean_val = np.mean(val_accs)
                std_train = np.std(train_accs)
                std_val = np.std(val_accs)
                
                # Plot with error bands
                ax.fill_between(folds, 
                               [mean_train - std_train] * len(folds),
                               [mean_train + std_train] * len(folds),
                               color=colors['train'], alpha=0.1)
                ax.fill_between(folds, 
                               [mean_val - std_val] * len(folds),
                               [mean_val + std_val] * len(folds),
                               color=colors['val'], alpha=0.1)
                
                # Plot lines
                ax.plot(folds, train_accs, 'o-', color=colors['train'], 
                       linewidth=2.5, markersize=9, label='Train', alpha=0.9,
                       markeredgewidth=1.5, markeredgecolor='white')
                ax.plot(folds, val_accs, 's-', color=colors['val'], 
                       linewidth=2.5, markersize=9, label='Validation', alpha=0.9,
                       markeredgewidth=1.5, markeredgecolor='white')
                
                # Add mean lines
                ax.axhline(y=mean_train, color=colors['train'], linestyle='--', 
                          alpha=0.4, linewidth=1.5, zorder=1)
                ax.axhline(y=mean_val, color=colors['val'], linestyle='--', 
                          alpha=0.4, linewidth=1.5, zorder=1)
                
                # Add value labels on points
                for i, (fold, train_acc, val_acc) in enumerate(zip(folds, train_accs, val_accs)):
                    # Only show every other label to avoid clutter
                    if i % 2 == 0 or len(folds) <= 5:
                        ax.text(fold, train_acc, f'{train_acc:.1f}', 
                               ha='center', va='bottom', fontsize=7, color=colors['train'],
                               fontweight='bold')
                        ax.text(fold, val_acc, f'{val_acc:.1f}', 
                               ha='center', va='top', fontsize=7, color=colors['val'],
                               fontweight='bold')
                
                # Stats box
                stats_text = f'Train: {mean_train:.2f}±{std_train:.2f}%\nVal: {mean_val:.2f}±{std_val:.2f}%'
                ax.text(0.02, 0.98, stats_text, 
                       transform=ax.transAxes, ha='left', va='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='white', 
                                            alpha=0.8, edgecolor='gray', linewidth=1))
                
                # Styling
                ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
                ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
                ax.set_title(f'{model_type} - {feature}', 
                            fontsize=12, fontweight='bold', pad=10)
                ax.set_xticks(folds)
                ax.set_xticklabels([f'{f}' for f in folds])
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
                ax.legend(loc='lower right', fontsize=9, framealpha=0.95, 
                         edgecolor='gray', fancybox=True)
                
                # Set y-axis limits for better comparison
                all_vals = train_accs + val_accs
                y_min = min(all_vals) - 3
                y_max = max(all_vals) + 3
                ax.set_ylim([max(0, y_min), min(100, y_max)])
                
                # Add subtle background
                ax.set_facecolor('#f9f9f9')
                
            except FileNotFoundError as e:
                ax.text(0.5, 0.5, f'Data Not Found\n{feature}', 
                       ha='center', va='center',
                       transform=ax.transAxes, fontsize=11, color='red')
                ax.set_title(f'{model_type} - {feature}', 
                            fontsize=12, fontweight='bold', pad=10)
                print(f"Warning: File not found for {model_type} {arch_name} {feature}")
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    # Save figure
    output_path = output_dir / f'architecture_comparison_{arch_name.lower().replace("-", "_")}_folds.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

def create_all_figures():
    """Create figures for all architectures"""
    print("="*60)
    print("Creating Architecture Comparison Figures")
    print("="*60)
    
    for arch in architectures:
        print(f"\nProcessing {arch}...")
        try:
            create_architecture_figure(arch)
        except Exception as e:
            print(f"✗ Error creating figure for {arch}: {str(e)}")
    
    print("\n" + "="*60)
    print("All figures created successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    create_all_figures()