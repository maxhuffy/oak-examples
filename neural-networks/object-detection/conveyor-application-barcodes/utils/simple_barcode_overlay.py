import cv2
import numpy as np
import depthai as dai
from threading import Lock
import time


class SimpleBarcodeOverlay(dai.node.HostNode):
    """
    Simple overlay that shows the most recently decoded barcode on the video stream and detection boxes.
    """
    
    def __init__(self):
        super().__init__()
        
        # Store multiple barcode texts with timestamps
        self.barcodes = {}  # {barcode_text: timestamp}
        self.latest_detections = None  # Store latest detections
        self.lock = Lock()
        self.max_display_time = 3.0  # Show barcodes for 3 seconds
        self.max_barcodes = 5  # Maximum number of barcodes to display at once
        self.input_barcode = self.createInput()

    def build(self, barcode_source: dai.Node.Output, video_source: dai.Node.Output, detection_source: dai.Node.Output = None) -> "SimpleBarcodeOverlay":
        
        self.link_args(video_source, detection_source)
        barcode_source.link(self.input_barcode)

        self.sendProcessingToPipeline(True)
        return self
    
    def process(self, video_msg: dai.ImgFrame, detection_msg: dai.Buffer = None) -> dai.ImgFrame:
        # Check for new barcode data (non-blocking) - Process ALL available messages
        try:
            with self.lock:
                current_time = time.time()
                # Process all available barcode messages to prevent queue backup
                while self.input_barcode.has():
                    barcode_msg = self.input_barcode.get()
                    data = barcode_msg.getData()
                    
                    if isinstance(data, bytes):
                        barcode_text = data.decode("utf-8", errors="ignore")
                    elif hasattr(data, 'tobytes'):
                        # If it's a numpy array, convert to bytes first
                        barcode_text = data.tobytes().decode("utf-8", errors="ignore")
                    else:
                        # Try to convert to string directly
                        barcode_text = str(data)
                    
                    # Add/update this barcode with current timestamp
                    self.barcodes[barcode_text] = current_time
                
                # Clean up old barcodes
                self.barcodes = {
                    code: timestamp for code, timestamp in self.barcodes.items()
                    if current_time - timestamp < self.max_display_time
                }
                
                # Limit number of displayed barcodes
                if len(self.barcodes) > self.max_barcodes:
                    # Keep only the most recent ones
                    sorted_barcodes = sorted(self.barcodes.items(), key=lambda x: x[1], reverse=True)
                    self.barcodes = dict(sorted_barcodes[:self.max_barcodes])
        except Exception as e:
            pass  # Silently handle barcode data errors
        
        # Check for new detection data (non-blocking)
        try:
            
            with self.lock:
                self.latest_detections = detection_msg
        except Exception as e:
            pass  # Silently handle detection data errors
        
        # Get video frame (blocking)
        try:
                
            # Get the frame as OpenCV format
            frame = video_msg.getCvFrame()
            
            # Draw detection boxes and barcodes
            with self.lock:
                current_time = time.time()
                
                # Draw detection boxes if available
                if self.latest_detections:
                    self._draw_detection_boxes(frame, self.latest_detections)
                
                # Clean up expired barcodes
                active_barcodes = {
                    code: timestamp for code, timestamp in self.barcodes.items()
                    if current_time - timestamp < self.max_display_time
                }
                self.barcodes = active_barcodes
                
                if active_barcodes:
                    self._draw_multiple_barcodes(frame, active_barcodes, current_time)
            
            # Create output message
            output_msg = dai.ImgFrame()
            output_msg.setData(frame)
            output_msg.setTimestamp(video_msg.getTimestamp())
            output_msg.setSequenceNum(video_msg.getSequenceNum())
            output_msg.setWidth(frame.shape[1])
            output_msg.setHeight(frame.shape[0])
            output_msg.setType(video_msg.getType())
            
            cv2.imshow("Barcode Detection", output_msg.getCvFrame())
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                exit(0)
            
        except Exception as e:
            pass  # Silently handle frame processing errors
            

    def _draw_multiple_barcodes(self, frame, active_barcodes, current_time):
        """Draw multiple barcode texts with prettier styling"""
        try:
            h, w = frame.shape[:2]
            
            # Sort barcodes by timestamp (newest first)
            sorted_barcodes = sorted(active_barcodes.items(), key=lambda x: x[1], reverse=True)
            
            # Panel configuration
            panel_width = int(w * 0.35)  # 35% of frame width
            panel_x = w - panel_width - 20  # Right side with margin
            panel_y_start = 20
            
            # Draw main panel background
            panel_height = min(len(sorted_barcodes) * 80 + 40, h - 40)
            cv2.rectangle(frame, 
                        (panel_x - 15, panel_y_start), 
                        (panel_x + panel_width + 15, panel_y_start + panel_height),
                        (0, 0, 0), -1)  # Black background
            
            # Draw panel border with gradient effect
            cv2.rectangle(frame, 
                        (panel_x - 15, panel_y_start), 
                        (panel_x + panel_width + 15, panel_y_start + panel_height),
                        (100, 200, 255), 3)  # Orange border
            
            # Draw panel header
            header_text = f"Detected Barcodes ({len(sorted_barcodes)})"
            cv2.putText(frame, header_text, (panel_x, panel_y_start + 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
            
            # Draw separator line
            cv2.line(frame, (panel_x, panel_y_start + 35), 
                    (panel_x + panel_width, panel_y_start + 35), 
                    (100, 200, 255), 2)
            
            # Draw each barcode with age-based styling
            for i, (barcode_text, timestamp) in enumerate(sorted_barcodes):
                if i >= self.max_barcodes:
                    break
                    
                y_pos = panel_y_start + 60 + (i * 70)
                if y_pos > h - 50:  # Don't draw outside frame
                    break
                
                # Calculate age and opacity
                age = current_time - timestamp
                alpha = max(0.3, 1.0 - (age / self.max_display_time))
                
                # Color coding based on age
                if age < 1.0:  # Fresh (green)
                    color = (0, 255, 0)
                    bg_color = (0, 50, 0)
                elif age < 2.0:  # Medium (yellow)
                    color = (0, 255, 255)
                    bg_color = (0, 50, 50)
                else:  # Old (red)
                    color = (0, 150, 255)
                    bg_color = (0, 25, 50)
                
                # Apply alpha
                color = tuple(int(c * alpha) for c in color)
                
                # Draw individual barcode background
                cv2.rectangle(frame, 
                            (panel_x, y_pos - 25), 
                            (panel_x + panel_width, y_pos + 15),
                            bg_color, -1)
                
                # Draw barcode text
                display_text = barcode_text
                if len(display_text) > 18:  # Truncate long barcodes
                    display_text = display_text[:15] + "..."
                
                cv2.putText(frame, display_text, (panel_x + 10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Draw age indicator
                age_text = f"{age:.1f}s"
                cv2.putText(frame, age_text, (panel_x + panel_width - 50, y_pos - 8), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                # Draw freshness indicator (dot)
                dot_color = (0, 255, 0) if age < 1.0 else (0, 255, 255) if age < 2.0 else (0, 150, 255)
                cv2.circle(frame, (panel_x + 5, y_pos - 5), 4, dot_color, -1)
            
        except Exception as e:
            pass  # Silently handle barcode drawing errors
    
    def _draw_detection_boxes(self, frame, detections):
        """Draw detection bounding boxes on the frame"""
        try:
            h, w = frame.shape[:2]
            
            for detection in detections.detections:
                # Get normalized coordinates from rotated_rect
                xmin, ymin, xmax, ymax = detection.rotated_rect.getOuterRect()
                
                # Convert to pixel coordinates
                x1 = int(xmin * w)
                y1 = int(ymin * h)
                x2 = int(xmax * w)
                y2 = int(ymax * h)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(0, min(w-1, x2))
                y2 = max(0, min(h-1, y2))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence score
                confidence = detection.confidence
                label = f"Barcode: {confidence:.2f}"
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
        except Exception as e:
            pass  # Silently handle detection box drawing errors
