import rosbag
from rospy import Time
import rospy
import os

def modify_timestamps(input_bag_path, output_bag_path, target_topics, time_offset):
    # Open the input bag file
    with rosbag.Bag(input_bag_path, 'r') as input_bag:
        # Open the output bag file
        with rosbag.Bag(output_bag_path, 'w') as output_bag:
            for topic, msg, t in input_bag.read_messages():
                if topic in target_topics:
                    # Modify the header timestamp
                    msg.header.stamp = t

                # Write the message to the output bag file
                output_bag.write(topic, msg, t)

if __name__ == "__main__":
    input_folder = '/media/khha/INTENSO/20240606-Testfahrt_IR_CAM/'  # Path to the folder containing input bag files
    target_topics = ['/basler/stereo_left_anon', '/basler/stereo_right_anon']  # List of topics to modify
    time_offset = rospy.Duration(secs=20, nsecs=0)  # Time offset to add to header timestamps

    # List all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.bag'):
            input_bag_path = os.path.join(input_folder, file_name)
            output_bag_path = os.path.join(input_folder, file_name.replace('.bag', '_fixed.bag'))
            modify_timestamps(input_bag_path, output_bag_path, target_topics, time_offset)