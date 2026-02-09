#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def parse_hybrid_dna():
    project_data = {}
    vvp_dir = 'vvp_outputs/atomic'
    raw_dir = './verilog_proj'
    
    # 1. Try to get Synthesis DNA (The most accurate)
    if os.path.exists(vvp_dir):
        for f in os.listdir(vvp_dir):
            proj_name = f.split('__')[0]
            if proj_name not in project_data: project_data[proj_name] = Counter()
            
            with open(os.path.join(vvp_dir, f), 'r') as file:
                content = file.read()
                gates = re.findall(r'\.functor\s+([\w/]+)', content)
                for g in gates: project_data[proj_name][f"gate_{g}"] += 1

    # 2. Textual Fallback (If synthesis failed, find OR/NOT gates in code)
    # This specifically looks for your "OR -> NOT" pattern in raw text
    for root, dirs, files in os.walk(raw_dir):
        proj_name = root.split(os.sep)[1] if len(root.split(os.sep)) > 1 else ""
        if not proj_name or proj_name == "verilog_proj": continue
        
        if proj_name not in project_data: project_data[proj_name] = Counter()
        
        for f in files:
            if f.endswith(('.v', '.sv')):
                with open(os.path.join(root, f), 'r', errors='ignore') as code:
                    text = code.read()
                    # Count instances of NOT (~, !), OR (|, ||), and assignments
                    project_data[proj_name]['raw_NOT'] += text.count('~') + text.count('!')
                    project_data[proj_name]['raw_OR'] += text.count('|')
                    project_data[proj_name]['raw_AND'] += text.count('&')
                    # Look for the specific "OR -> NOT" pattern: ~(...|...)
                    project_data[proj_name]['pattern_NOR'] += len(re.findall(r'~\s*\(.*\|.*\)', text))

    # Calculate Similarity
    df = pd.DataFrame(project_data).fillna(0).T
    sim = cosine_similarity(df)
    sim_df = pd.DataFrame(sim * 100, index=df.index, columns=df.index).round(2)
    
    sim_df.to_csv('final_similarity_report.csv')
    print("✅ Success! Similarity Matrix generated.")
    print(sim_df)

if __name__ == "__main__":
    parse_hybrid_dna()