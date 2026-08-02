import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
   
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
       
        self.PERSON_CLASS_ID = 0
        
    def detect_persons(self, image):
        
       
        results = self.model(image, verbose=False)
        
        detections = []
        
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID
                class_id = int(box.cls[0])
                
                # Check if it's a person and confidence is above threshold
                confidence = float(box.conf[0])
                
                if class_id == self.PERSON_CLASS_ID and confidence >= self.confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Calculate center
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'center': [cx, cy],
                        'confidence': confidence,
                        'area': area
                    })
        
        return detections
    
    def draw_detections(self, image, detections, gps_coords=None):
        
        vis_image = image.copy()
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            
            label = f"Person {confidence:.2f}"
            
          
            if gps_coords and idx < len(gps_coords):
                lat, lon = gps_coords[idx]
                if lat is not None and lon is not None:
                    label += f"\nLat: {lat:.6f}\nLon: {lon:.6f}"
            
           
            label_lines = label.split('\n')
            y_offset = y1 - 10
            
            for line in label_lines:
                (label_w, label_h), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis_image, 
                    (x1, y_offset - label_h - 5), 
                    (x1 + label_w, y_offset), 
                    (0, 255, 0), 
                    -1
                )
                cv2.putText(
                    vis_image, 
                    line, 
                    (x1, y_offset - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 0), 
                    1
                )
                y_offset -= (label_h + 5)
            
            # Draw center point
            cx, cy = det['center']
            cv2.circle(vis_image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        
        return vis_image
