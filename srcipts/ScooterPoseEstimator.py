import numpy as np
from geometry_msgs.msg import Pose  # Assuming you are using ROS
from tf.transformations import quaternion_from_euler
from sensor_msgs.msg import NavSatFix
from geodesy import utm
import csv

global altitude
global lat
global lon
global Location_in_UTM
global angle

class ScooterPoseEstimator:
    def __init__(self, HFOV, VFOV, IMAGE_WIDTH, IMAGE_HEIGHT):
        self.set_camera_parameters(HFOV, VFOV, IMAGE_WIDTH, IMAGE_HEIGHT)

    def set_camera_parameters(self, HFOV, VFOV, IMAGE_WIDTH, IMAGE_HEIGHT):
        self.HFOV = HFOV
        self.VFOV = VFOV
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT

    def get_pose(self, bbox, class_name):
        orient = self.pose_from_ratio(w = bbox[2], h = bbox[3])
        object_height = physical_dimensions[class_name][orient]
        distance = self.get_distance(bbox[3], object_height)

        pose = Pose()
        pose.position.x = 0
        pose.position.y = 0
        pose.position.z = distance
        if orient == "stand":
            q = quaternion_from_euler(0, 0, 0, 'syxz')
        else:
            q = quaternion_from_euler(0, 0, np.deg2rad(90), 'syxz')
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose
    
    def get_nav_sat_location(self, bbox, distance_to_vehicle, camera, header, class_name):
        actual_position_in_utm = Location_in_UTM
        actual_angle_to_utm = angle  # angle in radians. 0 is North
        
        angle_to_vehicle = self.get_angle(x = bbox[0], width = bbox[2], camera = camera)
        # Missing offset calculation
        # Replace these placeholders with actual calculations
        offset_east = 0
        offset_north = 0

        # Offset calculation from Ladybug to midpoint (SBG reference system)
        # Wegstrecke_in_east += offset_east
        # Wegstrecke_in_north += offset_north

        # Convert bin position to vehicle UTM coordinates and combine
        Wegstrecke_in_north = distance_to_vehicle.cpu().numpy() * np.cos(
            np.deg2rad(360 - angle_to_vehicle.cpu().numpy()) - actual_angle_to_utm)
        Wegstrecke_in_east = -distance_to_vehicle.cpu().numpy() * np.sin(
            np.deg2rad(360 - angle_to_vehicle.cpu().numpy()) - actual_angle_to_utm)


        Object_location = utm.UTMPoint()
        Object_location.easting = actual_position_in_utm.easting + Wegstrecke_in_east
        Object_location.northing = actual_position_in_utm.northing + Wegstrecke_in_north
        Object_location.altitude = actual_position_in_utm.altitude
        Object_location.zone = actual_position_in_utm.zone
        Object_location.band = actual_position_in_utm.band
        Object_location = Object_location.toMsg()

        # Create message with the bin position
        Nachricht = NavSatFix()
        Nachricht.header = header
        Nachricht.latitude = Object_location.latitude
        Nachricht.longitude = Object_location.longitude
        Nachricht.altitude = Object_location.altitude
        # Log bin position for evaluation
        with open('/home/IBEO.AS/khha/catkin_ws/data/scooter_positions.csv', 'a', newline='') as csv_file:
            data_writer = csv.writer(csv_file)
            data_writer.writerow([class_name, Nachricht.latitude, Nachricht.longitude,Nachricht.altitude])
        return Nachricht
        
    def pose_from_ratio(self, w, h):
        print(f"ration= {h/w}")
        if h/w > 0.6:
            return 'stand'
        else:
            return 'laying'

    def get_distance(self, pixel_height, object_height):
        distance = ((object_height / pixel_height) * (self.IMAGE_HEIGHT / 2)) / np.tan(np.deg2rad(self.VFOV / 2))
        return distance

    def GPS_callback(self, data): #Aktuelle Position des Fahrzeugs ermitteln und zwischenspeichern
        global lat
        lat = data.latitude
        global lon
        lon = data.longitude
        global altitude
        altitude = data.altitude
        global Location_in_UTM
        Location_in_UTM = utm.fromLatLong(lat, lon, altitude)

    def Euler_callback(self, data): # Winkel des Fahrzeugs Richtung Norden zwischenspeichern
        global angle
        angle = data.angle.z # data angles are in rad

    def get_angle(self, x, width, camera): # Übernommen aus der Masterarbeit YanLing, mit Abwandelungen von Maximilian Weltz
        # IMAGE_WIDTH = 2047
        # HFOV = 101.662896
        middlePixelPosition = width*0.5 + x
        if camera == 0:
            offset_a = 0
            offset_b = 360
        elif camera == 1:
            offset_a = 72
            offset_b = 72
        elif camera == 2:
            offset_a = 144
            offset_b = 144
        elif camera == 3:
            offset_a = 216
            offset_b = 216
        elif camera == 4:
            offset_a = 288
            offset_b = 288
        if(middlePixelPosition >= 1023.5):
            angle = offset_a + ((middlePixelPosition - 1023.5) * self.HFOV) / self.IMAGE_WIDTH
        else:
            angle = offset_b - ((1023.5 - middlePixelPosition) * self.HFOV) / self.IMAGE_WIDTH
        return ((angle+11.9)%360) # should correct miss-alignment of camera #11.9

physical_dimensions = {}
physical_dimensions['scooter_tier'] = {'stand':1.20,'laying':0.40}
physical_dimensions['scooter_bolt'] = {'stand':1.19,'laying':0.40}
physical_dimensions['scooter_lime'] = {'stand':1.20,'laying':0.40}
physical_dimensions['scooter_voi'] = {'stand':1.18,'laying':0.40}