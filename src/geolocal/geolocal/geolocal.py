import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu, NavSatFix

from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd

from ament_index_python.packages import get_package_share_directory
import os



class Geolocalization(Node):

    def __init__(self):
        super().__init__('Geolocalization')
        
        self.declare_parameter('confidence', 0.001)
        self.declare_parameter('output_csv', 'detections.csv')

        self.conf_thresh = self.get_parameter('confidence').value
        self.output_csv = self.get_parameter('output_csv').value

        pkg_share = get_package_share_directory('geolocal')
        self.model_path = os.path.join(pkg_share, 'models', 'best1.pt')
      
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.pan = 0.0
        self.tilt = 0.0
        self.yaw = 0.0
        self.roll = 0.0  
        self.lat = 0
        self.lon = 0

        self.frame_id = 0
        self.csv_rows = []

        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.base_altitude = None

        self.detections_this_frame = [] 

        self.intrinsic_matrix = np.array([[554.3, 0.0, 320.080],
                                       [0.0, 554.3, 240.482],
                                      [0.0, 0.0, 1.0]])
        

        self.sub = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.image_callback, 
            10
        )
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.camera_imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data', 
            self.imu_callback, 
            qos
        )

        self.global_position_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.pos_callback,
            qos
        )

        self.timer = self.create_timer(0.1, self.main_callback)


    def quat_to_euler(self, w, x, y, z):
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(2 * (w * y - z * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return roll, pitch, yaw


    def pos_callback(self, msg):
        
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        
        if self.base_altitude is None: 
            self.base_altitude = msg.altitude
        
        #self.altitude = msg.altitude - self.base_altitude
        self.altitude = 10 
    
    def imu_callback(self, msg):

        q = msg.orientation

        self.pan, self.tilt, self.roll = self.quat_to_euler(q.w, q.x, q.y, q.z)


    def image_callback(self, msg):

        self.frame_id += 1

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        results = self.model.track(
            frame,
            persist=True,
            classes=[0], 
            conf=self.conf_thresh,
            verbose=False
        )

        self.detections_this_frame = [] 

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, self.track_id in zip(boxes, ids):

                x1, y1, x2, y2 = map(int, box)
                
                self.detections_this_frame = [x1, y1, x2, y2]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"ID {self.track_id} , Lat {float(self.lat):.7f} , Lon {float(self.lon):.7f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(frame, f"Lat: {self.latitude}, Lon: {self.longitude}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO Drone View", frame)
        
        cv2.waitKey(1)
    

    def local_to_global(self, x, y, lat, lon):
        
        R = 6378137.0  # Earth radius (WGS84) in meters

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        dlat = y / R
        dlon = x / (R * np.cos(lat_rad))

        lat = lat_rad + dlat
        lon = lon_rad + dlon

        return np.rad2deg(lat), np.rad2deg(lon)


    def main_callback(self):

        if self.altitude is None:
            return

        if not self.detections_this_frame:
            return

        R_Pan = np.array([[np.cos(self.pan), -np.sin(self.pan), 0],[np.sin(self.pan), np.cos(self.pan), 0],[0,0,1]])
        R_Tilt = np.array([[np.cos(self.tilt), 0, np.sin(self.tilt)], [0,1,0], [-np.sin(self.tilt), 0, np.cos(self.tilt)]])
        R_Roll = np.array([[1,0,0], [0, np.cos(self.roll), -np.sin(self.roll)], [0, np.sin(self.roll), np.cos(self.roll)]])

        R_ENU_NED = np.diag([1, -1, -1])

        R_Body_NED = R_Pan @ R_Tilt @ R_Roll

        R_Body_ENU = R_ENU_NED @ R_Body_NED

        K_inverse = np.linalg.inv(self.intrinsic_matrix)

        R_Body_Cam = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

        P = np.array([[(self.detections_this_frame[0] + self.detections_this_frame[2]) / 2],
                                        [(self.detections_this_frame[1] + self.detections_this_frame[3]) / 2],
                                        [1]]
                                    )

        d_cam  = K_inverse @ P

        d_body = R_Body_Cam @ d_cam

        if d_body[2] <= 0:
            return

        t = self.altitude / d_body[2]

        p_body = t * d_body

        p_ENU = R_Body_ENU @ p_body

        x = float(p_ENU[0])
        y = float(p_ENU[1])

        self.get_logger().info(f"ID: {self.track_id}")

        self.get_logger().info(f"Local == X: {x}, Y: {y}")

        lat, lon = self.local_to_global(-x, -y, self.latitude, self.longitude)

        self.get_logger().info(f"Global == X: {lat:.7f}, Y: {lon:.7f}")
        self.get_logger().info(f"Drone == X: {self.latitude:.7f}, Y: {self.longitude:.7f}")


    def destroy_node(self):
        df = pd.DataFrame(self.csv_rows)
        df.to_csv(self.output_csv, index=False)
        self.get_logger().info(f"✅ Saved detections to {self.output_csv}")
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args = None):

    rclpy.init(args = args)

    node = Geolocalization()
    
    try:
        rclpy.spin(node)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        node.destroy_node()
        rclpy.shutdown()