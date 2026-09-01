# escooter_ros

A ROS 1 package that detects e-scooters in vehicle camera images with YOLOv8, estimates
each scooter's distance and orientation from its bounding box, and converts that into a
geographic position using the vehicle's GNSS/INS data.

Developed as an elective project at HAW Hamburg (Faculty of Engineering and Computer
Science, Department of Information and Electrical Engineering), 2024.

## How it works

The pipeline runs in one node and has three stages:

1. **Detection.** A YOLOv8 model (Ultralytics) runs on incoming camera frames and returns
   bounding boxes, class labels, and confidence scores for four scooter brands.
2. **Pose estimation.** `ScooterPoseEstimator` classifies each detection as *standing* or
   *laying* from its bounding-box aspect ratio (ratio > 0.6 means standing), looks up the
   corresponding real-world height, and derives the distance from the camera:

   ```
   distance = (object_height / pixel_height) * (IMAGE_HEIGHT / 2) / tan(VFOV / 2)
   ```

3. **Localization.** The vehicle's position (`SbgEkfNav`) and heading (`SbgEkfEuler`) are
   converted to UTM. The scooter's bearing relative to the camera, combined with the
   estimated distance, gives its UTM position, which is published back as a `NavSatFix`.

Known scooter classes and their assumed physical heights, in metres:

| Class | Standing | Laying |
|---|---|---|
| `scooter_tier` | 1.20 | 0.40 |
| `scooter_bolt` | 1.19 | 0.40 |
| `scooter_lime` | 1.20 | 0.40 |
| `scooter_voi` | 1.18 | 0.40 |

## Requirements

- ROS 1 Noetic, Ubuntu 20.04
- Python 3.8+

ROS packages (see `package.xml`): `cv_bridge`, `geometry_msgs`, `image_geometry`,
`image_transport`, `message_filters`, `sensor_msgs`, `std_msgs`, `tf2_ros`, `vision_msgs`,
`visualization_msgs`.

Not declared in `package.xml` but imported at runtime, so install them separately:

- `sbg_driver` — provides `SbgEkfNav` and `SbgEkfEuler`
- `geodesy` — UTM conversion (`sudo apt install ros-noetic-geodesy`)
- `ros_numpy` (`sudo apt install ros-noetic-ros-numpy`)
- `ultralytics`, `opencv-python`, `numpy` (`pip install ultralytics opencv-python`)
- `scikit-learn`, `matplotlib` — only for the offline analysis scripts

## Build

```bash
cd ~/catkin_ws/src
git clone https://github.com/hamdykhfagy/escooter_ros.git
cd ~/catkin_ws
catkin build          # or: catkin_make
source devel/setup.bash
```

## Run

```bash
roslaunch escooter_ros escooter.launch
```

Then play a recorded bag in a second terminal:

```bash
rosbag play ./data/1.bag --clock --rate 1.0
```

`start.sh` launches the node and the bag player together in two terminal tabs, and
`close.sh` shuts them both down. Both assume the workspace is at `~/catkin_ws`.
`rosbag_play_script.sh` replays a directory of bags in sequence — edit `BAGS_DIRECTORY`
at the top before using it.

## Topics

**Subscribed**

| Topic | Type | Notes |
|---|---|---|
| `~topic_name_cam` | `sensor_msgs/Image` | Primary camera |
| `~topic_name_cam2` | `sensor_msgs/Image` | Second camera, used when `cam2_logic` is `or`/`and` |
| `/sbg/ekf_nav` | `sbg_driver/SbgEkfNav` | Vehicle GNSS position |
| `/sbg/ekf_euler` | `sbg_driver/SbgEkfEuler` | Vehicle heading |
| `/velodyne_points` | `sensor_msgs/PointCloud2` | Only when `log_results` is true |

**Published**

| Topic | Type | Notes |
|---|---|---|
| `~topic_name_result` | `escooter_ros/EscooterResult` | Detections, annotated image, and positions |
| `~topic_name_result_image` | `sensor_msgs/Image` | Annotated frame |
| `escooter/posess` | `geometry_msgs/Pose` | One message per detection |

`EscooterResult` bundles a `Header`, a `vision_msgs/Detection2DArray`, an annotated
`Image`, and a `NavSatFix[]` of estimated scooter positions. It also declares a
`sensor_msgs/Image[] masks` field, which the node does not currently populate.

## Camera modes

`cam2_logic` selects how the two cameras are combined:

- **`or`** — either camera's frames trigger the pipeline independently.
- **`and`** — both cameras must produce frames, synchronized with
  `ApproximateTimeSynchronizer`, and detection counts must match.
- **anything else (e.g. `off`)** — single camera. If `log_results` is true, camera frames
  are instead synchronized against the LiDAR point cloud for offline evaluation.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `~topic_name_cam` | `/right_cam_node/image_raw` | Primary camera topic |
| `~topic_name_cam2` | `/left_cam_node/image_raw` | Secondary camera topic |
| `~topic_name_result` | `/escooter/result` | Result topic |
| `~topic_name_result_image` | `/escooter/image` | Annotated image topic |
| `~yolo_model` | `best_June.pt` | Weights file under `models/` |
| `~yolo_conf_thresh` | `0.5` | Detection confidence threshold |
| `~cam2_logic` | `or` | `or`, `and`, or off — see above |
| `~CAMERA` | `2` | Ladybug camera index (0–4), sets the bearing offset |
| `~HFOV` | `101.662896` | Horizontal field of view, degrees |
| `~VFOV` | `111.383104` | Vertical field of view, degrees |
| `~IMAGE_WIDTH` | `2047` | Input image width, pixels |
| `~IMAGE_HEIGHT` | `2464` | Input image height, pixels |
| `~min_width` / `~max_width` | `10` / `10000` | Bounding-box width filter, pixels |
| `~min_height` / `~max_height` | `10` / `10000` | Bounding-box height filter, pixels |
| `~log_results` | `false` | Save frames and point clouds for offline analysis |
| `~log_csv_path` | `<package>/data/scooter_positions.csv` | Position log; set empty to disable |
| `~debug` | `rviz` | `rviz` or `cv_image` |

The launch file overrides several of these — notably `yolo_conf_thresh` to `0.6` and the
bounding-box filters to 30/400 — so check `launch/escooter.launch` for the values that
actually apply to a launched run.

## Logging

Two independent logging paths:

- **`~log_csv_path`** appends one row per detected scooter — class, latitude, longitude,
  altitude. It defaults to `data/scooter_positions.csv` inside the package and creates
  the directory if needed. Set the parameter to an empty string to turn it off.
- **`~log_results`** saves the annotated frame and the synchronized LiDAR point cloud per
  sequence number, for offline distance validation against LiDAR ground truth.

## Models

Pre-trained weights in `models/`:

- `best_June.pt` — default, trained on the project's annotated scooter dataset
- `v3_best.pt` — earlier revision
- `yolov8n.pt` — stock YOLOv8-nano baseline

## Repository layout

```
launch/escooter.launch      Node launch file with all parameters
models/                     YOLOv8 weights
msg/EscooterResult.msg      Custom detection + localization message
rviz/escooter.rviz          RViz configuration
scripts/
  escooter.py               Main ROS node
  ScooterPoseEstimator.py   Distance, orientation, and UTM localization
  dbscan.py                 DBSCAN clustering of logged point clouds
  bag_editors/              Bag timestamp repair and batch replay
  tutorials/                Standalone experiments, not part of the node
start.sh / close.sh         Start and stop node + bag player together
rosbag_play_script.sh       Sequential bag replay
```

Everything under `scripts/tutorials/` is exploratory work kept for reference. Those files
contain hardcoded absolute paths from the original development machines and will not run
unedited.

## Known limitations

- `CMakeLists.txt` has no `catkin_install_python` rule, so the node is found through the
  source tree rather than an installed location. It works with `roslaunch`, but the
  package does not install cleanly.
- `launch/escooter.launch` passes `result_image_topic`, while the node reads
  `~topic_name_result_image`. That parameter currently has no effect from the launch file.
- Distance estimation assumes the scooter's full height is visible in the bounding box;
  partial occlusion inflates the estimated distance.

## Attribution

The bearing (`get_angle`) and distance (`get_distance`) helpers are adapted from an
earlier master's thesis at the same department.
