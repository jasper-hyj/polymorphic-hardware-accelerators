import json
import sys
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def extract_dna():
    fingerprints = {}
    # Look for the JSON netlists created by the Bash script
    for f in os.listdir('fingerprints'):
        if f.endswith('.json'):
            with open(f'fingerprints/{f}') as jf:
                try:
                    data = json.load(jf)
                    counts = {}
                    # We count every logic gate and port bit
                    for mod in data['modules'].values():
                        # Count Gates
                        for cell in mod['cells'].values():
                            ctype = cell['type']
                            counts[ctype] = counts.get(ctype, 0) + 1
                        # Count Port Bits
                        for port in mod['ports'].values():
                            counts['port_bits'] = counts.get('port_bits', 0) + len(port['bits'])
                    
                    fingerprints[f.replace('.json', '')] = counts
                except:
                    continue

    if not fingerprints:
        print("No fingerprints found!")
        return

    # Create the 100% similarity matrix
    df = pd.DataFrame(fingerprints).fillna(0).T
    sim = cosine_similarity(df)
    sim_df = pd.DataFrame(sim * 100, index=df.index, columns=df.index).round(2)
    
    sim_df.to_csv('logic_similarity_report.csv')
    print("✅ Success! 100% scale similarity matrix saved to logic_similarity_report.csv")

if __name__ == "__main__":
    extract_dna()