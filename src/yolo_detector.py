import os

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE

# Detection is run on a centred crop of the frame only. 0.8 keeps the central
# 80% of the width and 80% of the height; anything outside is never fed to the
# model. Boxes are mapped back to full-frame pixel coordinates afterwards, so
# the downstream pixel -> GPS projection is unaffected.
DEFAULT_ROI_FRACTION = 0.8


class YOLODetector:

    # Open-vocabulary prompt for YOLOE models. Listing only 'person' makes the
    # model's class 0 the person class, matching PERSON_CLASS_ID below.
    YOLOE_CLASSES = ['person']

    PERSON_CLASS_ID = 0

    def __init__(self, model_name='yoloe-26l-seg.pt', confidence_threshold=0.5,
                 roi_fraction=DEFAULT_ROI_FRACTION):

        self.confidence_threshold = confidence_threshold
        self.roi_fraction = float(roi_fraction)

        if 'yoloe' in os.path.basename(str(model_name)).lower():
            # YOLOE is open-vocabulary: it needs a text prompt before it will
            # predict anything.
            self.model = YOLOE(model_name)
            self.model.set_classes(
                self.YOLOE_CLASSES,
                self.model.get_text_pe(self.YOLOE_CLASSES),
            )
        else:
            self.model = YOLO(model_name)

    def roi_bounds(self, image_shape):
        """Pixel bounds (x1, y1, x2, y2) of the centred detection ROI."""
        h, w = image_shape[:2]
        frac = min(max(self.roi_fraction, 0.0), 1.0)

        roi_w = max(1, int(round(w * frac)))
        roi_h = max(1, int(round(h * frac)))

        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2

        return x1, y1, x1 + roi_w, y1 + roi_h

    def detect_persons(self, image):

        # Only the central ROI is given to the detector.
        rx1, ry1, rx2, ry2 = self.roi_bounds(image.shape)
        roi = np.ascontiguousarray(image[ry1:ry2, rx1:rx2])

        results = self.model(roi, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get class ID
                class_id = int(box.cls[0])

                # Check if it's a person and confidence is above threshold
                confidence = float(box.conf[0])

                if class_id == self.PERSON_CLASS_ID and confidence >= self.confidence_threshold:
                    # Get bounding box coordinates (ROI-local) and shift them
                    # back into full-frame coordinates.
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 += rx1
                    x2 += rx1
                    y1 += ry1
                    y2 += ry1

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

        # Outline the region detections are actually run on.
        rx1, ry1, rx2, ry2 = self.roi_bounds(vis_image.shape)
        cv2.rectangle(vis_image, (rx1, ry1), (rx2 - 1, ry2 - 1), (255, 200, 0), 1)
        cv2.putText(
            vis_image,
            f"ROI {int(round(self.roi_fraction * 100))}%",
            (rx1 + 4, ry1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1
        )

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
