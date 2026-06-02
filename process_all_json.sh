#!/bin/bash
# Script to apply add_bezier2coco.py to all JSON files in a folder

DATASET_FOLDER="datasets/forbin_dataset"
INPUT_PATTERN="*.json"

# Counter for processed files
count=0

# Iterate through all JSON files in the dataset folder
for input_file in "$DATASET_FOLDER"/$INPUT_PATTERN; do
    # Check if file exists (in case no files match)
    if [ -f "$input_file" ]; then
        
        echo "Processing: $input_file -> $input_file"
        
        # Run the Python script
        python utilities/add_bezier2coco.py "$input_file" "$input_file"

        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed $input_file"
            ((count++))
        else
            echo "✗ Error processing $input_file"
        fi
        
        echo ""
    fi
done

echo "Total files processed: $count"
