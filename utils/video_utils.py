import cv2
import time
import numpy as np
from typing import Tuple
from config.config import Config

class FPSCounter:
    """Calculate and track FPS."""
    
    def __init__(self, avg_over=30):
        self.frame_times = []
        self.avg_over = avg_over
        self. last_time = time.time()
    
    def update(self) -> float:
        """Update FPS calculation."""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self. last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.avg_over:
            self.frame_times.pop(0)
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return fps

def draw_face_box(frame: np.ndarray, 
                  face_location: Tuple[int, int, int, int],
                  name: str, 
                  confidence: float):
    """
    Draw bounding box and label on face.
    
    Args:
        frame: Image frame
        face_location: (top, right, bottom, left)
        name: Person's name
        confidence: Recognition confidence
    """
    top, right, bottom, left = face_location
    
    # Color based on recognition
    if name == "Unknown":
        color = (0, 0, 255)  # Red for unknown
    else:
        color = (0, 255, 0)  # Green for recognized
    
    # Draw box
    cv2.rectangle(frame, (left, top), (right, bottom), color, Config.BOX_THICKNESS)
    
    # Prepare label
    if Config.DISPLAY_CONFIDENCE:
        label = f"{name} ({confidence:.2f})"
    else:
        label = name
    
    # Draw label background
    label_height = 30
    cv2.rectangle(
        frame, 
        (left, bottom), 
        (right, bottom + label_height), 
        color, 
        cv2.FILLED
    )
    
    # Draw label text
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(
        frame, 
        label, 
        (left + 6, bottom + 20), 
        font, 
        0.6, 
        (255, 255, 255), 
        1
    )

def apply_blur_to_face(frame: np.ndarray, 
                       face_location: Tuple[int, int, int, int]) -> np.ndarray:
    """Apply blur to a face region."""
    top, right, bottom, left = face_location
    
    # Extract face region
    face = frame[top:bottom, left:right]
    
    # Apply Gaussian blur
    blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
    
    # Replace face region with blurred version
    frame[top:bottom, left:right] = blurred_face
    
    return frame