import cv2
import time
from collections import deque
import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.face_recognizer import FaceRecognizer
from config.config import Config
from loguru import logger
from utils.video_utils import FPSCounter, draw_face_box
from utils.database import RecognitionDatabase

class RealtimeFaceRecognition:
    """
    Real-time face recognition from webcam or video file.
    """
    
    def __init__(self, video_source=Config.VIDEO_SOURCE):
        self.video_source = video_source
        self.recognizer = FaceRecognizer()
        self.fps_counter = FPSCounter()
        self.db = RecognitionDatabase()
        
        # Frame processing settings
        self.frame_skip = Config.FRAME_SKIP
        self.frame_count = 0
        
        # Smoothing for recognition results
        self.name_history = deque(maxlen=10)
        
        # Cache last recognition results for drawing between frames
        self.cached_results = []
        
        logger.info(f"Initialized real-time recognition with source: {video_source}")
    
    def start(self):
        """Start the real-time recognition system."""
        # Open video source
        video_capture = cv2.VideoCapture(self.video_source)
        
        if not video_capture.isOpened():
            logger.error(f"Could not open video source: {self. video_source}")
            return
        
        # Set video properties
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        logger.info("Starting real-time face recognition.  Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = video_capture. read()
                
                if not ret:
                    logger.warning("Failed to grab frame")
                    break
                
                self.frame_count += 1
                
                # Process every Nth frame for better performance
                if self.frame_count % self.frame_skip == 0:
                    # Recognize faces with aggressive scaling
                    self.cached_results = self.recognizer.recognize_faces(
                        frame, 
                        scale_factor=0.25  # More aggressive scaling for speed
                    )
                
                # Draw cached results on every frame for smooth display
                for face_location, name, confidence in self.cached_results:
                    # Draw bounding box and label
                    draw_face_box(
                        frame, 
                        face_location, 
                        name, 
                        confidence
                    )
                    
                    # Log recognition event (only when processing new frame)
                    if self.frame_count % self.frame_skip == 0 and name != "Unknown":
                        self.db.log_recognition(name, confidence)
                
                # Display FPS
                fps = self. fps_counter.update()
                if Config. DISPLAY_FPS:
                    cv2.putText(
                        frame,
                        f"FPS:  {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # Display the frame
                cv2.imshow('Real-time Face Recognition', frame)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit command received")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
        
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            logger.info("Real-time recognition stopped")

def main():
    """Main function for real-time recognition."""
    logger.add(Config.LOG_FILE, rotation="10 MB")
    
    print("=" * 60)
    print("Face Recognition System - Real-time Recognition")
    print("=" * 60)
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("\n" + "=" * 60)
    
    recognition = RealtimeFaceRecognition()
    recognition.start()

if __name__ == "__main__":
    main()