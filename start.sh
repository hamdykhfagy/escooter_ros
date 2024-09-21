#!/bin/bash

cd ~/catkin_ws
source devel/setup.bash

# Command 1: Replace with your first command
gnome-terminal --tab --title="Escooter_ros Node" -- bash -c "roslaunch escooter_ros escooter.launch debug:='rviz'; sleep 10"
sleep 3

# Command 2: Replace with your second command
gnome-terminal --tab --title="Bag Runner" -- bash -c "./src/escooter_ros/rosbag_play_script.sh; sleep 5"
