#!/bin/bash
mkdir -p vvp_out

for dir in ./verilog_proj/*; do
    if [ -d "$dir" ]; then
        proj_name=$(basename "$dir")
        echo "🔍 Scavenging DNA from $proj_name..."
        
        # Create a temp file to store all gate info for this project
        touch "vvp_out/${proj_name}.combined_vvp"

        # Find every Verilog file
        find "$dir" -name "*.v" -o -name "*.sv" | while read -r file; do
            # Attempt to compile individual file to assembly format
            # We use -E to just preprocess if full compilation fails
            iverilog -g2012 -o "temp.vvp" "$file" 2>/dev/null
            
            if [ -s "temp.vvp" ]; then
                cat "temp.vvp" >> "vvp_out/${proj_name}.combined_vvp"
            else
                # Fallback: Just extract raw keywords if synthesis fails
                echo "Fallback for $(basename "$file")"
            fi
        done
        rm -f temp.vvp
    fi
done