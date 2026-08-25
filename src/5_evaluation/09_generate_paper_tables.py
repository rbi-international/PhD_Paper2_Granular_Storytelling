import pandas as pd
import os

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_tables():
    print("--- Generating Final Results Table for PhD Paper ---")
    
    # 1. Define the experimental results manually (based on your runs)
    data = {
        "Model Architecture": [
            "GPT-2 Medium (Baseline)", 
            "GPT-2 Medium + Steering", 
            "GPT-2 Large (Optimized)", 
            "GPT-2 Large + Hybrid Steering"
        ],
        "Parameters": ["355M", "355M", "774M", "774M"],
        "Technique": [
            "LoRA (Rank 8)", 
            "LoRA + Lexical Boost", 
            "LoRA (Instruction Tuned)", 
            "LoRA + Instruction + Steering"
        ],
        "Top-1 Accuracy": ["6.67%", "20.00%", "13.12%", "47.50%"],
        "Joy F1": ["0.07", "0.53", "0.16", "0.73"],
        "Fear F1": ["0.17", "0.59", "0.08", "0.82"],
        "Sadness F1": ["0.20", "0.17", "0.17", "0.70"],
        "Inference Speed (est)": ["Fast", "Medium", "Slow", "Medium"]
    }
    
    df = pd.read_csv("evaluation_results_instruction.csv") if os.path.exists("evaluation_results_instruction.csv") else None
    
    # Create DataFrame
    results_df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = os.path.join(PROJECT_ROOT, "results", "comparisons", "final_paper_table.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print("\n--- TABLE I: MODEL PERFORMANCE COMPARISON ---")
    print(results_df.to_string(index=False))
    print(f"\nSaved to: {output_path}")
    print("\nYou can copy this table directly into your IEEE/Journal Paper.")

if __name__ == "__main__":
    generate_tables()