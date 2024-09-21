#!/bin/bash

# Function to close the terminals
close_terminals() {
    echo "Closing terminals..."

    # Find and kill the Escooter_ros Node terminal
    PID1=$(pgrep -f "roslaunch escooter_ros escooter.launch debug:='rviz'")
    if [ -n "$PID1" ]; then
        kill $PID1
        echo "Closed Escooter_ros Node terminal"
    else
        echo "Escooter_ros Node terminal not found"
    fi

    # Find and kill the Bag Runner terminal
    PID2=$(pgrep -f "./src/escooter_ros/rosbag_play_script.sh")
    if [ -n "$PID2" ]; then
        kill $PID2
        echo "Closed Bag Runner terminal"
    else
        echo "Bag Runner terminal not found"
    fi
}

# Call the function to close the terminals
close_terminals
