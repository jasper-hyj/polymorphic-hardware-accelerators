import os
import re
import pandas as pd
import numpy as np # Added for filling NaNs
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def extract_logic_fingerprint(file_path):
    try:
        with open(file_path, 'r', errors='ignore') as f:
            code = f.read()
    except Exception:
        return Counter()
        
    stats = Counter()
    # 1. Sequential Logic
    stats['seq_blocks'] = len(re.findall(r'always\s*@\s*\(posedge', code))
    
    # 2. Combinational Gates
    stats['gate_not'] = code.count('~') + code.count('!')
    stats['gate_or'] = code.count('|')
    stats['gate_and'] = code.count('&')
    stats['gate_xor'] = code.count('^')
    
    # 3. Target Pattern: OR -> NOT
    stats['pattern_nor'] = len(re.findall(r'[~!]\s*\([^|&]*\|[^|&]*\)', code))
    
    # 4. Data Flow
    stats['assigns'] = len(re.findall(r'\bassign\b', code))
    
    return stats

def run_analysis(root_dir):
    project_results = {}
    
    if not os.path.exists(root_dir):
        print(f"❌ Directory {root_dir} not found!")
        return

    for subdir in os.listdir(root_dir):
        path = os.path.join(root_dir, subdir)
        if not os.path.isdir(path): continue
        
        print(f"🧬 Scanning DNA: {subdir}...")
        project_total = Counter()
        
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(('.v', '.sv')):
                    file_dna = extract_logic_fingerprint(os.path.join(root, f))
                    project_total.update(file_dna)
        
        project_results[subdir] = project_total

    # Create DataFrame
    df = pd.DataFrame(project_results).fillna(0).T
    
    # --- FIX STARTS HERE ---
    # Normalize by project size, then fill NaNs (from division by zero) with 0
    df_norm = df.div(df.sum(axis=1), axis=0).fillna(0) 
    
    # Ensure we don't pass an empty or all-zero dataframe to cosine_similarity
    if df_norm.empty or df_norm.sum().sum() == 0:
        print("❌ Error: No valid hardware logic was found in any project folder.")
        return
    # --- FIX ENDS HERE ---

    # Calculate Similarity
    sim = cosine_similarity(df_norm)
    sim_df = pd.DataFrame(sim * 100, index=df.index, columns=df.index).round(2)
    
    sim_df.to_csv('hardware_similarity_report.csv')
    print("\n✅ Analysis Complete! Results saved to hardware_similarity_report.csv")
    print(sim_df)

if __name__ == "__main__":
    # Ensure the path matches your environment
    run_analysis('./verilog_proj')