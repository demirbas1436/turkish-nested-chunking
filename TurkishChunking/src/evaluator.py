import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import os

def evaluate_and_plot(y_true, y_pred, target_name, output_dir):
    # Flatten for traditional metrics
    y_true_flat = [tag for sent in y_true for tag in sent]
    y_pred_flat = [tag for sent in y_pred for tag in sent]
    
    # Filter out empty or underscore tags if necessary (though they are classes here)
    labels = sorted(list(set(y_true_flat + y_pred_flat)))
    
    print(f"\n--- {target_name} Evaluation ---")
    print(classification_report(y_true_flat, y_pred_flat, labels=labels, zero_division=0))
    
    acc = accuracy_score(y_true_flat, y_pred_flat)
    print(f"Accuracy: {acc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f'Confusion Matrix - {target_name}')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'cm_{target_name}.png'))
    plt.close()

    # Save metrics to CSV for report
    report = classification_report(y_true_flat, y_pred_flat, labels=labels, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_dir, f'metrics_{target_name}.csv'))
    
    return acc, report
