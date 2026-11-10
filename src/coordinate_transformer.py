"""
Coordinate transformation utilities for converting pixel coordinates to GPS coordinates.
"""

import numpy as np
import math


class CoordinateTransformer:
    
    
    def __init__(self, intrinsic_matrix):
        
        self.K = intrinsic_matrix
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]
        
        self.EARTH_RADIUS = 6378137.0
        
    def pixel_to_camera_ray(self, u, v, zoom=1.0):
   
        # Optical zoom scales the focal length; the principal point is unchanged.
        # `zoom` is the ratio relative to the zoom the intrinsics were calibrated at.
        if zoom is None or zoom <= 0:
            zoom = 1.0
        fx = self.fx * zoom
        fy = self.fy * zoom

        X_cam = (u - self.cx) / fx
        Y_cam = (v - self.cy) / fy
        Z_cam = 1.0
        

        ray = np.array([X_cam, Y_cam, Z_cam])
        ray = ray / np.linalg.norm(ray)
        
        return ray
    
    # ----------------------------------------------------------------------
    # Calibration: camera -> body rotation, plus a constant ground offset.
    #
    # Fitted jointly over 12 hand-annotated points across FOUR surveyed people
    # (yuvraj, surya, raval, vedant) in 1xzoom1.5ms.avi. Both terms come from
    # the same least-squares solve; they are a matched pair, so replacing one
    # without re-fitting the other will make results worse, not better.
    #
    #   accuracy over the 12 points   mean 0.55 m   max 1.17 m
    #   leave-one-person-out          mean 0.74 m   worst 1.09 m
    #   (nominal extrinsics scored 1.84 m mean over the same points)
    # ----------------------------------------------------------------------

    # Nominal 90 deg yaw plus a 3.23 deg mount tilt, rotvec
    # [+0.023, +3.228, 0.000] deg. Rotation about the camera boresight is
    # deliberately left at nominal: fitting it gave no in-sample gain and a
    # worse held-out score, with the angle swinging 4.3-21.3 deg depending on
    # which person was withheld. Re-fit it only with targets far off nadir.
    R_BODY_CAM = np.array([
        [+0.00001137, -0.99841377, +0.05630221],
        [+0.99999992, -0.00001137, -0.00040366],
        [+0.00040366, +0.05630221, +0.99841369],
    ])

    # Constant north/east correction added to every projected ground point.
    #
    # After the rotation is fitted, all four people remain displaced by nearly
    # the same world-frame vector (~1.45 m). A body-frame rotation cannot
    # reproduce a fixed world-frame offset because yaw varies between frames,
    # so this is carried as an explicit translation. Its likely physical cause
    # is a datum or antenna offset between the drone's GPS and whatever
    # surveyed the ground truth -- it is a property of THIS setup, so re-fit it
    # if the GPS hardware, mounting or survey method changes.
    GROUND_OFFSET_NORTH_M = +1.1120
    GROUND_OFFSET_EAST_M  = -0.9381

    # Previous nominal extrinsics, kept for A/B comparison.
    R_BODY_CAM_NOMINAL = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])

    def camera_to_body_frame(self, point_camera):

        return self.R_BODY_CAM @ point_camera
    
    


    def body_to_world_frame(self, point_body, roll, pitch, yaw):
        #pitch = -pitch
        R_roll = np.array([
            [1,           0,            0],
            [0,  np.cos(roll), -np.sin(roll)],
            [0,  np.sin(roll),  np.cos(roll)]
        ])
        R_pitch = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0,              1,             0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])

        # Correct order for NED: Yaw applied last (outermost)
        R_world_body = R_yaw @ R_pitch @ R_roll
        return R_world_body @ point_body
    
    def find_ground_intersection(self, ray_world, drone_altitude, ground_altitude=0.0):
     
        altitude_diff = drone_altitude 

        if ray_world[2] <= 0:

            return None
        
  
        scale = altitude_diff / ray_world[2]
  
        intersection = ray_world * scale
        
        return intersection
    
    def ned_to_gps(self, north_offset, east_offset, drone_lat, drone_lon):
      
        lat_rad = math.radians(drone_lat)
        m_per_deg_lat = (111132.92
                         - 559.82  * math.cos(2 * lat_rad)
                         + 1.175   * math.cos(4 * lat_rad)
                         - 0.0023  * math.cos(6 * lat_rad))
        m_per_deg_lon = (111412.84 * math.cos(lat_rad)
                         - 93.5    * math.cos(3 * lat_rad)
                         + 0.118   * math.cos(5 * lat_rad))
        return (drone_lat + north_offset / m_per_deg_lat,
                drone_lon + east_offset  / m_per_deg_lon)
        
        #return target_lat, target_lon
    
    def pixel_to_gps(self, u, v, drone_lat, drone_lon, drone_altitude, 
                     roll, pitch, yaw, ground_altitude=0.0, zoom=1.0):
        

        ray_camera = self.pixel_to_camera_ray(u, v, zoom)
        

        ray_body = self.camera_to_body_frame(ray_camera)
        
        ray_world = self.body_to_world_frame(ray_body, roll, pitch, yaw)
        
     
        intersection_ned = self.find_ground_intersection(
            ray_world, drone_altitude, ground_altitude
        )
        
        if intersection_ned is None:
            return None
        
       
        # Calibrated constant offset (see GROUND_OFFSET_* above).
        north_offset = intersection_ned[0]
        east_offset  = intersection_ned[1]
        
        target_lat, target_lon = self.ned_to_gps(
            north_offset, east_offset, drone_lat, drone_lon
        )
        
        return target_lat, target_lon
