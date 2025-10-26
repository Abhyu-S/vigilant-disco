"""
Metric computation utilities
"""
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np


def compute_accuracy(outputs, labels):
    """Compute accuracy"""
    _, predictions = torch.max(outputs, 1)
    correct = (predictions == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def compute_f1(outputs, labels, num_classes=100):
    """Compute F1 score"""
    _, predictions = torch.max(outputs, 1)
    # Move to CPU for sklearn
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    # Compute F1 for each class then average
    f1 = f1_score(labels_np, predictions_np, average='weighted', zero_division=0)
    return f1


def compute_confusion_matrix(outputs, labels, num_classes=100):
    """Compute confusion matrix"""
    _, predictions = torch.max(outputs, 1)
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    cm = confusion_matrix(labels_np, predictions_np, labels=list(range(num_classes)))
    return cm

