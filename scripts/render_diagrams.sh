#!/bin/bash

# This script renders all Mermaid diagrams in the ../diagrams directory into PNG images.

# --- Configuration ---
DIAGRAMS_DIR="../diagrams"
OUTPUT_DIR="../images/diagrams"
MMDC_COMMAND="mmdc"

# --- Pre-flight Check ---
# Check if mmdc is installed
if ! command -v $MMDC_COMMAND &> /dev/null
then
    echo "Error: The Mermaid CLI command '$MMDC_COMMAND' could not be found."
    echo "Please install it globally by running:"
    echo "npm install -g @mermaid-js/mermaid-cli"
    exit 1
fi

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# --- Main Loop ---
echo "Searching for .mmd files in $DIAGRAMS_DIR..."

for diagram_file in "$DIAGRAMS_DIR"/*.mmd; do
    if [ -f "$diagram_file" ]; then
        filename=$(basename -- "$diagram_file")
        base_filename="${filename%.*}"
        output_file="$OUTPUT_DIR/${base_filename}.png"

        echo "Rendering $filename to $output_file..."

        # Execute the render command
        $MMDC_COMMAND -i "$diagram_file" -o "$output_file" --backgroundColor transparent --scale 4 -p puppeteer-config.json
        
        if [ $? -eq 0 ]; then
            echo "Successfully rendered $output_file"
        else
            echo "Error rendering $filename"
        fi
    fi
done

echo "Diagram rendering complete."
