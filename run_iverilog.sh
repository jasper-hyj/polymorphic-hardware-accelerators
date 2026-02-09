#!/bin/bash
TARGET_DIR="./verilog_proj"
# Clean and recreate the directory
rm -rf vvp_outputs/atomic
mkdir -p vvp_outputs/atomic

echo "🚀 Starting High-Compatibility Atomic Extraction..."

find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    proj=$(basename "$dir")
    echo "  🔨 Analyzing Project: $proj"
    
    # Find every verilog/systemverilog file
    find "$dir" -type f \( -name "*.v" -o -name "*.sv" \) -not -path "*/Vivado/*" | while read -r file; do
        # Create a safe filename for the output
        safe_fname=$(echo "$file" | tr '/' '_' | tr '.' '_')
        
        # -g2012: Enables SystemVerilog support
        # -t null: Flattens logic without requiring a full testbench/simulation
        iverilog -g2012 -t null -o "vvp_outputs/atomic/${proj}__${safe_fname}.vvp" "$file" > /dev/null 2>&1
    done

    # Count how many files actually succeeded for this project
    success_count=$(ls vvp_outputs/atomic/${proj}__* 2>/dev/null | wc -l)
    if [ "$success_count" -gt 0 ]; then
        echo "    ✅ Captured DNA for $success_count files."
    else
        echo "    ❌ Failed to extract any DNA. (Iverilog cannot parse this project's syntax)"
    fi
done

# Verification
total_files=$(ls vvp_outputs/atomic | wc -l)
echo "📊 Total DNA segments captured: $total_files"