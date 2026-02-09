import os
import re
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def diagnostic_scanner(file_path):
    with open(file_path, 'r', errors='ignore') as f:
        code = f.read()
    
    # Remove comments and normalize whitespace
    code = re.sub(r'//.*|/\*.*?\*/', '', code, flags=re.DOTALL)
    clean_code = " ".join(code.split())
    
    stats = Counter()
    # Broad strokes: Keywords and Operators
    stats['ops'] = clean_code.count('~') + clean_code.count('|') + clean_code.count('&') + clean_code.count('^')
    stats['assigns'] = len(re.findall(r'\bassign\b', clean_code))
    stats['always'] = len(re.findall(r'\balways\b', clean_code))
    # Connection: The specific OR -> NOT pattern you want
    stats['nor_pattern'] = len(re.findall(r'~[ ]*\([^|]*\|[^|]*\)', clean_code))
    
    return stats

def run_diagnostic(root_dir):
    project_results = {}
    for subdir in os.listdir(root_dir):
        path = os.path.join(root_dir, subdir)
        if not os.path.isdir(path): continue
        
        project_total = Counter()
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(('.v', '.sv')):
                    project_total.update(diagnostic_scanner(os.path.join(root, f)))
        
        # DEBUG: Print what we found for each project
        print(f"📁 {subdir}: {dict(project_total)}")
        project_results[subdir] = project_total

    df = pd.DataFrame(project_results).fillna(0).T
    # Critical Fix: Add a tiny epsilon to avoid division by zero
    df_norm = df.div(df.sum(axis=1) + 1e-9, axis=0)
    
    sim = cosine_similarity(df_norm)
    sim_df = pd.DataFrame(sim * 100, index=df.index, columns=df.index).round(2)
    print("\n✅ New Similarity Matrix:")
    print(sim_df)

if __name__ == "__main__":
    run_diagnostic('./verilog_proj')