#!/bin/bash

# Define the directory containing bag files
BAGS_DIRECTORY="/media/khha/INTENSO/20240606-Testfahrt_IR_CAM/has_scooter"

# Define the pattern for bag files
BAG_FILE_PATTERN="*_{32..34}.bag"

# Define the patterns for bag files
# PATTERNS=$(eval echo "$BAGS_DIRECTORY/*_fixed.bag")
PATTERNS=$(eval echo "$BAGS_DIRECTORY/*_{11..39}.bag")

# Combine directory and pattern to form the full path to bag files
BAG_FILES_PATH="$BAGS_DIRECTORY/$BAG_FILE_PATTERN"

rosparam set use_sim_time true 
# EXPLICIT_BAG_FILE="/media/khha/INTENSO/20240606-Testfahrt_IR_CAM/has_scooter/tesla_2024-06-06-15-53-42_11_fixed.bag"
# echo "Playing $EXPLICIT_BAG_FILE"
# rosbag play $EXPLICIT_BAG_FILE --clock --rate 1.0 -l

# Loop through all bag files in the specified directory
for bagfile in "$BAGS_DIRECTORY/"*.bag; do
    if [ -f "$bagfile" ]; then
        echo "Playing $bagfile"
        rosbag play "$bagfile" --clock --rate 1
    else
        echo "No matching .bag files found for the pattern $bagfile"
    fi
done