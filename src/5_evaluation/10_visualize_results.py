import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Input Data Path (From your Hybrid Run)
HYBRID_RESULTS_PATH = "hybrid_results.csv" 

# --- 1. Model Comparison Bar Chart (The "Progression" Plot) ---
def plot_comparison():
    print("Generating Model Comparison Plot...")
    
    # Data from your specific experiments
    models = [
        "GPT-2 Medium\n(Baseline)", 
        "GPT-2 Medium\n(+Steering)", 
        "GPT-2 Large\n(Optimized)", 
        "GPT-2 Large\n(Hybrid)"
    ]
    accuracy = [6.67, 20.00, 13.12, 47.50] # The exact numbers you achieved
    colors = ['#bdc3c7', '#95a5a6', '#7f8c8d', '#2ecc71'] # Grey for baselines, Green for Hybrid

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracy, color=colors, edgecolor='black', alpha=0.8)
    
    # Add numbers on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.title("Impact of Model Size vs. Neuro-Symbolic Steering\non Emotional Arc Adherence", fontsize=14, fontweight='bold')
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
    plt.ylim(0, 60)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save
    save_path = os.path.join(RESULTS_DIR, "Figure1_Model_Comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

# --- 2. Confusion Matrix (The "Granular" Plot) ---
def plot_confusion_matrix():
    print("Generating Confusion Matrix...")
    
    if not os.path.exists(HYBRID_RESULTS_PATH):
        print(f"Warning: {HYBRID_RESULTS_PATH} not found. Skipping CM.")
        return

    df = pd.read_csv(HYBRID_RESULTS_PATH)
    
    # Get labels
    labels = sorted(list(set(df['Target'].unique()) | set(df['Detected'].unique())))
    # Ensure specific order if possible (Plutchik pairs)
    desired_order = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
    # Filter only labels that exist in the data to avoid crashes
    labels = [l for l in desired_order if l in labels] + [l for l in labels if l not in desired_order and l != "Neutral"]
    
    # Compute Matrix
    cm = confusion_matrix(df['Target'], df['Detected'], labels=labels, normalize='true') # Normalize to show percentages

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, square=True, cbar_kws={'label': 'Adherence Probability'})
    
    plt.title("Confusion Matrix: GPT-2 Large + Hybrid Steering", fontsize=14, fontweight='bold')
    plt.xlabel("Detected Emotion (RoBERTa Teacher)", fontsize=12)
    plt.ylabel("Target Emotion (Prompt)", fontsize=12)
    plt.xticks(rotation=45)
    
    # Save
    save_path = os.path.join(RESULTS_DIR, "Figure2_Confusion_Matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

# --- 3. F1 Score Breakdown (The "Success" Plot) ---
def plot_f1_scores():
    print("Generating F1 Score Plot...")
    
    # Data from your Hybrid Run
    emotions = ["Fear", "Joy", "Sadness", "Anger", "Surprise", "Trust", "Anticipation"]
    f1_scores = [0.82, 0.73, 0.70, 0.46, 0.36, 0.34, 0.16] # Sorted by success
    
    plt.figure(figsize=(10, 6))
    plt.barh(emotions, f1_scores, color='#3498db', edgecolor='black')
    plt.xlabel("F1-Score (Precision vs Recall Balance)", fontsize=12)
    plt.title("Per-Emotion Performance (Hybrid Model)", fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Acceptable Baseline')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Save
    save_path = os.path.join(RESULTS_DIR, "Figure3_Emotion_F1.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_comparison()
    plot_confusion_matrix()
    plot_f1_scores()
    print("\n--- All Plots Generated Successfully ---")