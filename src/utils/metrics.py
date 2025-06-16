"""
Evaluation metrics for the model.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

def calculate_category_metrics(predictions: List[int], 
                             labels: List[int], 
                             class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-category metrics.
    
    Args:
        predictions: List of predicted class indices
        labels: List of true class indices
        class_names: List of class names
        
    Returns:
        Dict containing per-category metrics
    """
    metrics = {}
    
    for i, class_name in enumerate(class_names):
        # Calculate true positives, false positives, false negatives
        tp = sum((p == i) & (l == i) for p, l in zip(predictions, labels))
        fp = sum((p == i) & (l != i) for p, l in zip(predictions, labels))
        fn = sum((p != i) & (l == i) for p, l in zip(predictions, labels))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': sum(l == i for l in labels)
        }
    
    return metrics

def calculate_confusion_matrix(predictions: List[int], 
                             labels: List[int], 
                             num_classes: int) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        predictions: List of predicted class indices
        labels: List of true class indices
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for p, l in zip(predictions, labels):
        cm[l][p] += 1
        
    return cm

def calculate_overall_metrics(predictions: List[int], 
                            labels: List[int]) -> Dict[str, float]:
    """
    Calculate overall metrics.
    
    Args:
        predictions: List of predicted class indices
        labels: List of true class indices
        
    Returns:
        Dict containing overall metrics
    """
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(labels)
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'total_samples': total
    }

def format_metrics_report(metrics: Dict[str, Dict[str, float]], 
                         overall_metrics: Dict[str, float]) -> str:
    """
    Format metrics into a readable report.
    
    Args:
        metrics: Per-category metrics
        overall_metrics: Overall metrics
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("\nMetrics Report:")
    report.append("=" * 50)
    
    # Overall metrics
    report.append("\nOverall Metrics:")
    report.append(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    report.append(f"Total Samples: {overall_metrics['total_samples']}")
    
    # Per-category metrics
    report.append("\nPer-Category Metrics:")
    report.append("-" * 50)
    report.append(f"{'Category':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    report.append("-" * 50)
    
    for category, cat_metrics in metrics.items():
        report.append(
            f"{category:<20} "
            f"{cat_metrics['precision']:.4f}    "
            f"{cat_metrics['recall']:.4f}    "
            f"{cat_metrics['f1_score']:.4f}    "
            f"{cat_metrics['support']}"
        )
    
    return "\n".join(report)