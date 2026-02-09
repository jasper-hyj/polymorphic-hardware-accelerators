import os
import re
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def extract_structural_fingerprint(proj_path):
    stats = Counter()
    
    # Regex patterns for hardware structures
    patterns = {
        'registers': r'\breg\b|\balways_ff\b',
        'comb_logic': r'\bassign\b|\balways_comb\b',
        'multipliers': r'\*',
        'mux_logic': r'\?|case|if',
        'bit_widths_high': r'\[(6[3-9]|[7-9]\d|1\d\d):0\]', # 64-bit or higher
        'bit_widths_med': r'\[(1[5-9]|2\d|3[0-1]):0\]',   # 16-bit to 32-bit
        'state_machines': r'parameter.*S\d+|next_state',
        'mem_arrays': r'reg.*\[.*\]\s+\w+\s+\[\d+\]'       # Memory arrays
    }

    for root, _, files in os.walk(proj_path):
        for f in files:
            if f.endswith(('.v', '.sv')):
                try:
                    with open(os.path.join(root, f), 'r', errors='ignore') as code_f:
                        content = code_f.read()
                        for feature, regex in patterns.items():
                            stats[feature] += len(re.findall(regex, content))
                        # Capture port count density
                        stats['io_ports'] += len(re.findall(r'\b(input|output)\b', content))
                except:
                    continue
    return stats

def analyze_all():
    proj_root = './verilog_proj' # Update to your path
    all_features = {}

    for proj in os.listdir(proj_root):
        path = os.path.join(proj_root, proj)
        if os.path.isdir(path):
            print(f"🧬 Extracting DNA from {proj}...")
            all_features[proj] = extract_structural_fingerprint(path)

    # Convert to DataFrame and Normalize
    df = pd.DataFrame(all_features).fillna(0).T
    
    # Normalizing ensures that a large project isn't "dissimilar" to a small one
    # just because it has more lines. We want the RATIO of logic types.
    df_norm = df.div(df.sum(axis=1) + 1e-9, axis=0)
    
    # Calculate Cosine Similarity
    sim = cosine_similarity(df_norm)
    sim_df = pd.DataFrame(sim * 100, index=df.index, columns=df.index).round(2)
    
    print("\n--- Structural Similarity Matrix (%) ---")
    print(sim_df)
    sim_df.to_csv('final_similarity.csv')

if __name__ == "__main__":
    analyze_all()