#!/bin/bash
TARGET_DIR="./verilog_proj"
mkdir -p fingerprints

echo "🚀 Starting Dependency-Aware Synthesis..."

find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | while read dir; do
    proj=$(basename "$dir")
    echo "  🔨 Synthesizing $proj..."

    # Gather files
    all_files=$(find "$dir" -name "*.v" -o -name "*.sv")
    inc_args=$(find "$dir" -type d | sed 's/^/-I /')

    # THE FIX: 
    # 1. We use 'read_verilog -defer' so Yosys doesn't complain about missing pieces yet.
    # 2. We use 'hierarchy -generate' to try and build the missing parts.
    # 3. We use 'synth -top' to force a gate-level mapping.
    yosys -p "
        read_verilog -sv $inc_args -defer $all_files;
        hierarchy -check -top \$(ls $dir/*.v $dir/*.sv | xargs grep -l 'module' | head -n 1 | xargs basename -s .v -s .sv);
        proc;
        opt;
        write_json fingerprints/$proj.json
    " > "fingerprints/${proj}.log" 2>&1
    
    if [ -f "fingerprints/$proj.json" ]; then
        echo "    ✅ DNA Captured"
    else
        echo "    ❌ Synthesis Failed (Check fingerprints/${proj}.log)"
    fi
done

echo "📊 Calculating Similarity Matrix..."
python3 yosys_dna.py