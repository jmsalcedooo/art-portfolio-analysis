import json
import csv
import os
from pathlib import Path

# Base directory
base_dir = r"C:\Users\PC\OneDrive\Documents\ArtModelThesis\results"

# File configurations
configs = {
    'Local': [
        ('AlexNet', 'Brushstroke', 'local_alexnet_brushstroke.json'),
        ('AlexNet', 'Color', 'local_alexnet_color.json'),
        ('AlexNet', 'Texture', 'local_alexnet_texture.json'),
        ('EfficientNet-B3', 'Brushstroke', 'local_efficientnet_b3_brushstroke.json'),
        ('EfficientNet-B3', 'Color', 'local_efficientnet_b3_color.json'),
        ('EfficientNet-B3', 'Texture', 'local_efficientnet_b3_texture.json'),
        ('ResNet50', 'Brushstroke', 'local_resnet50_brushstroke.json'),
        ('ResNet50', 'Color', 'local_resnet50_color.json'),
        ('ResNet50', 'Texture', 'local_resnet50_texture.json'),
        ('VGG16', 'Brushstroke', 'local_vgg16_brushstroke.json'),
        ('VGG16', 'Color', 'local_vgg16_color.json'),
        ('VGG16', 'Texture', 'local_vgg16_texture.json'),
    ],
    'Global': [
        ('AlexNet', 'Brushstroke', 'global_alexnet_brushstroke.json'),
        ('AlexNet', 'Color', 'global_alexnet_color.json'),
        ('AlexNet', 'Texture', 'global_alexnet_texture.json'),
        ('EfficientNet-B3', 'Brushstroke', 'global_efficientnet_b3_brushstroke.json'),
        ('EfficientNet-B3', 'Color', 'global_efficientnet_b3_color.json'),
        ('EfficientNet-B3', 'Texture', 'global_efficientnet_b3_texture.json'),
        ('ResNet50', 'Brushstroke', 'global_resnet50_brushstroke.json'),
        ('ResNet50', 'Color', 'global_resnet50_color.json'),
        ('ResNet50', 'Texture', 'global_resnet50_texture.json'),
        ('VGG16', 'Brushstroke', 'global_vgg16_brushstroke.json'),
        ('VGG16', 'Color', 'global_vgg16_color.json'),
        ('VGG16', 'Texture', 'global_vgg16_texture.json'),
    ]
}

# Create summary CSV
summary_file = os.path.join(base_dir, 'architecture_comparison_summary.csv')
with open(summary_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model_Type', 'Architecture', 'Feature', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'Average', 'Std_Dev'])
    
    for model_type, config_list in configs.items():
        subdir = 'Local_Architecture_Comparison' if model_type == 'Local' else 'Global_Architecture_Comparison'
        
        for arch, feature, filename in config_list:
            filepath = os.path.join(base_dir, subdir, filename)
            
            with open(filepath, 'r') as jf:
                data = json.load(jf)
            
            fold_accs = []
            for i in range(1, 6):
                fold_key = f'fold_{i}'
                if fold_key in data:
                    best_val_acc = data[fold_key].get('best_val_acc', 0)
                    fold_accs.append(best_val_acc)
            
            avg = sum(fold_accs) / len(fold_accs) if fold_accs else 0
            std = (sum((x - avg) ** 2 for x in fold_accs) / len(fold_accs)) ** 0.5 if fold_accs else 0
            
            writer.writerow([model_type, arch, feature] + [f'{acc:.4f}' for acc in fold_accs] + [f'{avg:.4f}', f'{std:.4f}'])

print(f"Summary CSV created: {summary_file}")

# Create detailed history CSV
detailed_file = os.path.join(base_dir, 'architecture_comparison_detailed.csv')
with open(detailed_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model_Type', 'Architecture', 'Feature', 'Fold', 'Epoch', 
                     'Train_Acc', 'Train_Loss', 'Val_Acc', 'Val_Loss', 
                     'Precision', 'Recall', 'F1_Score'])
    
    for model_type, config_list in configs.items():
        subdir = 'Local_Architecture_Comparison' if model_type == 'Local' else 'Global_Architecture_Comparison'
        
        for arch, feature, filename in config_list:
            filepath = os.path.join(base_dir, subdir, filename)
            
            with open(filepath, 'r') as jf:
                data = json.load(jf)
            
            for i in range(1, 6):
                fold_key = f'fold_{i}'
                if fold_key in data and 'history' in data[fold_key]:
                    for epoch_data in data[fold_key]['history']:
                        writer.writerow([
                            model_type, arch, feature, i,
                            epoch_data.get('epoch', ''),
                            f"{epoch_data.get('train_acc', 0):.4f}",
                            f"{epoch_data.get('train_loss', 0):.4f}",
                            f"{epoch_data.get('val_acc', 0):.4f}",
                            f"{epoch_data.get('val_loss', 0):.4f}",
                            f"{epoch_data.get('precision', 0):.4f}",
                            f"{epoch_data.get('recall', 0):.4f}",
                            f"{epoch_data.get('f1_score', 0):.4f}"
                        ])

print(f"Detailed CSV created: {detailed_file}")
print("Done!")