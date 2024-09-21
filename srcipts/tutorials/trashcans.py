import rospy
import cv2
import numpy as np
import os
from geodesy import utm
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Image
from sbg_driver.msg import SbgEkfNav
from sbg_driver.msg import SbgEkfEuler
from ladybug6.msg import ladybug5_rectified_images
from cv_bridge import CvBridge
import csv
bridge = CvBridge()
# globale Variablen können innerhalb von Klassen verhindert werden, dadurch, dass Variablen die gesamte Objektlebensdauer erhalten bleiben
global altitude
global lat
global lon
global Location_in_UTM
global angle

# with open('/home/workstation3/Desktop/Muelltonnen_positionen.csv', 'w', newline='') as csvfile:
# datawriter = csv.writer(csvfile)
# datawriter.writerow(['header','lat','lon','alt'])

def detect(frame,cfg,weights,classes):
    # Neuronales Netz laden
    net = cv2.dnn.readNet(weights,cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    # Bild dimensionen ermitteln
    height, width, channels = frame.shape
    # Bild durch das neuronale Netz verarbeiten
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True,
    crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    # Schleife um alle Vorhersagen zu prüfen10. Anhang
    100
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.82: # Ist die Sicherheit einer Detektion > 82% -> Mülltonne erkannt
                # Mülltonnenposition ermitteln
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Bounding-Bos Koordinaten
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # gefundene Mülltonnen speichern
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences

def Rectified_Image_callback(data):
    # Store current position and angle
    actual_position_in_utm = Location_in_UTM
    actual_angle_to_utm = angle  # angle in radians. 0 is North

    # Process each Ladybug image
    for camera in range(5):
        if camera == 0:
            frame = data.camera0
        elif camera == 1:
            frame = data.camera1
        elif camera == 2:
            frame = data.camera2
        elif camera == 3:
            frame = data.camera3
        elif camera == 4:
            frame = data.camera4

        # Convert image from ROS format to OpenCV format
        frame = bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8")

        # Execute neural network
        boxes, confidences = detect(frame, cfg, weights, classes)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Process all detections
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]  # Bounding-Box Properties
                if h > 30:  # Minimum bounding box height
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(frame, "{}%".format(int(confidences[i] * 100)),
                                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

                    # Calculate angle of the bin to the vehicle's longitudinal axis,
                    # as well as the distance
                    angle_to_vehicle = get_angle(x, w, camera)
                    object_height = 1.2  # actual height of a trash bin is 0.8 m
                    distance_to_vehicle = get_distance(h, object_height)

                    # Missing offset calculation
                    # Replace these placeholders with actual calculations
                    offset_east = 0
                    offset_north = 0

                    # Offset calculation from Ladybug to midpoint (SBG reference system)
                    # Wegstrecke_in_east += offset_east
                    # Wegstrecke_in_north += offset_north

                    # Convert bin position to vehicle UTM coordinates and combine
                    Wegstrecke_in_north = distance_to_vehicle * np.cos(np.deg2rad(360 - angle_to_vehicle) - actual_angle_to_utm)
                    Wegstrecke_in_east = -distance_to_vehicle * np.sin(np.deg2rad(360 - angle_to_vehicle) - actual_angle_to_utm)
                    Object_location = utm.UTMPoint()
                    Object_location.easting = actual_position_in_utm.easting + Wegstrecke_in_east
                    Object_location.northing = actual_position_in_utm.northing + Wegstrecke_in_north
                    Object_location.altitude = actual_position_in_utm.altitude
                    Object_location.zone = actual_position_in_utm.zone
                    Object_location.band = actual_position_in_utm.band
                    Object_location = Object_location.toMsg()

                    # Create message with the bin position
                    Nachricht = NavSatFix()
                    Nachricht.header = data.header
                    Nachricht.latitude = Object_location.latitude
                    Nachricht.longitude = Object_location.longitude
                    Nachricht.altitude = Object_location.altitude

                    # Log bin position for evaluation
                    with open('/home/workstation3/Desktop/Muelltonnen_positionen.csv', 'a', newline='') as csvfile:
                        datawriter = csv.writer(csvfile)
                        datawriter.writerow([Nachricht.header, Nachricht.latitude, Nachricht.longitude,
                                             Nachricht.altitude])

                    # Publish bin position as NavSatFix
                    pub_pos.publish(Nachricht)

                    # Convert frame back to ROS Image format and publish
                    frame_copy = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    pub.publish(frame_copy)


def GPS_callback(data): #Aktuelle Position des Fahrzeugs ermitteln und zwischenspeichern
    global lat
    lat = data.latitude
    global lon
    lon = data.longitude
    global altitude
    altitude = data.altitude
    global Location_in_UTM
    Location_in_UTM = utm.fromLatLong(lat, lon, altitude)

def Euler_callback(data): # Winkel des Fahrzeugs Richtung Norden zwischenspeichern
    global angle
    angle = data.angle.z # data angles are in rad

def get_angle(x, width, camera): # Übernommen aus der Masterarbeit YanLing, mit Abwandelungen von Maximilian Weltz
    IMAGE_WIDTH = 2047
    HFOV = 101.662896
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
        angle = offset_a + ((middlePixelPosition - 1023.5) * HFOV) / IMAGE_WIDTH
    else:
        angle = offset_b - ((1023.5 - middlePixelPosition) * HFOV) / IMAGE_WIDTH
    return ((angle+11.9)%360) # should correct missalignment of camera #11.9

def get_distance(pixel_hight, object_hight): # Übernommen aus der Masterarbeit  YanLing, mit Abwandelungen von Maximilian Weltz
    # aus Arbeit von Yanling
    #define HFOV 101.662896
    VFOV = 111.383104
    #define IMAGE_WIDTH 2047
    #define IMAGE_HEIGHT 1231
    IMAGE_ORIGINAL_HEIGHT = 2464
    distance = ((object_hight/pixel_hight)*(IMAGE_ORIGINAL_HEIGHT/2))/np.tan(np.deg2rad(VFOV/2))
    return distance

if __name__ == '__main__':
    ## Load path to model config ##
    dirname = os.path.dirname(__file__)
    cfg = os.path.join('/home/workstation3/Downloads/yolov4-tiny-custom(4).cfg')
    weights = os.path.join('/home/workstation3/Downloads/yolov4-tiny-custom_best(4).weights')
    classes = ['bake']
    #Node initialisieren
    rospy.init_node("Muelleimer_detection_node")
    # Publisher & Subscriber erstellen
    pub = rospy.Publisher('/Muelleimer', Image, queue_size=1)
    sub = rospy.Subscriber("/ladybug/all_camera/image_rect_color", ladybug5_rectified_images, callback=Rectified_Image_callback)
    sub_gps = rospy.Subscriber("/sbg/ekf_nav",SbgEkfNav,callback=GPS_callback)
    sub_euler = rospy.Subscriber("/sbg/ekf_euler",SbgEkfEuler,callback=Euler_callback)
    pub_pos = rospy.Publisher('/Trash_Can_Detect_v2/Muelleimer_Location_msg', NavSatFix, queue_size=1)
    #Console info
    rospy.loginfo("Publisher & Subscriber has been started.")
    rospy.loginfo("Node has been started")
    rospy.spin()
    # Der code befindet sich in der Callback-funktion des Subscribers.
    # Immer wenn etwas empfangen wird, wird es verarbeitet und gepublished.