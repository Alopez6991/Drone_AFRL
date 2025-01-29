#!/bin/bash

# Script to convert all PNG files in a folder to a single PDF without transparency issues

# Directory containing PNG files (default to current directory)
input_directory="${1:-.}"

# Output PDF file
output_pdf="output_no_dmu_dt_no_flow.pdf"

# Create a temporary directory to store modified PNG files
temp_directory=$(mktemp -d)

# Remove alpha channel from all PNG files
for file in "$input_directory"/*.png; do
    convert "$file" -background white -alpha remove -alpha off "$temp_directory/$(basename "$file")"
done

# Find all PNG files in the temporary directory and convert to PDF
img2pdf "$temp_directory"/*.png -o "$output_pdf"

# Remove the temporary directory
rm -r "$temp_directory"

# Notify user
if [ $? -eq 0 ]; then
    echo "Successfully created $output_pdf from PNG files in $input_directory."
else
    echo "Failed to create PDF. Please make sure ImageMagick and img2pdf are installed and PNG files exist in the directory."
fi
