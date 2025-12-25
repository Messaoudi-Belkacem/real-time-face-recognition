import numpy as np
import cv2
import face_recognition
from typing import List, Tuple, Optional
from config.config import Config
from models.face_encoder import FaceEncoder
from models.face_detector import FaceDetector
from loguru import logger

class FaceRecognizer: 
    """
    Complete face recognition system combining detection and recognition.
    """
    
    def __init__(self):
        self.detector = FaceDetector()
        self.encoder = FaceEncoder()
        self.threshold = Config.RECOGNITION_THRESHOLD
        
        # Load known faces
        self.encoder.load_encodings()
        logger.info("Face recognizer initialized")
    
    def recognize_faces(self, image: np.ndarray, 
                       scale_factor: float = 1.0) -> List[Tuple[tuple, str, float]]:
        """
        Detect and recognize faces in an image. 
        
        Args:
            image: Input image (BGR format)
            scale_factor: Scale for faster processing
            
        Returns: 
            List of tuples (face_location, name, confidence)
        """
        # Convert to RGB for face_recognition library
        rgb_image = image[:, :, ::-1]
        
        # Scale down image for faster processing if requested
        if scale_factor != 1.0:
            height, width = rgb_image.shape[:2]
            rgb_image = cv2.resize(rgb_image, (int(width * scale_factor), int(height * scale_factor)))
        
        # Use face_recognition's built-in detection and encoding with num_jitters=0
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if len(face_locations) == 0:
            return []
        
        # Encode detected faces with num_jitters=0 to avoid dlib compatibility issues
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=0)
        
        # Scale locations back if we scaled the image
        if scale_factor != 1.0:
            face_locations = [(int(top / scale_factor), int(right / scale_factor), 
                             int(bottom / scale_factor), int(left / scale_factor))
                            for top, right, bottom, left in face_locations]
        
        results = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            name, confidence = self._match_face(face_encoding)
            results.append((face_location, name, confidence))
        
        return results
    
    def _match_face(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """
        Match a face encoding against known faces.
        
        Args:
            face_encoding: 128-dimensional face encoding
            
        Returns:
            Tuple of (name, confidence)
        """
        if len(self.encoder.known_encodings) == 0:
            return "Unknown", 0.0
        
        # Calculate distances to all known faces
        face_distances = face_recognition.face_distance(
            self.encoder.known_encodings, 
            face_encoding
        )
        
        # Find best match
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        # Convert distance to confidence score (0-1)
        confidence = 1.0 - best_distance
        
        if best_distance < self.threshold:
            name = self.encoder.known_names[best_match_index]
            return name, confidence
        else:
            return "Unknown", confidence
    
    def recognize_faces_batch(self, face_encodings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Recognize multiple faces at once.
        
        Args:
            face_encodings: List of face encodings
            
        Returns: 
            List of (name, confidence) tuples
        """
        results = []
        for face_encoding in face_encodings:
            name, confidence = self._match_face(face_encoding)
            results.append((name, confidence))
        
        return results
    
    def add_person(self, image: np.ndarray, name: str) -> bool:
        """
        Add a new person to the recognition database.
        
        Args:
            image: Image containing the person's face
            name: Person's name
            
        Returns: 
            True if successful, False otherwise
        """
        try:
            # Detect and encode face
            face_locations = self.detector.detect_faces(image)
            
            if len(face_locations) == 0:
                logger. warning(f"No face detected for {name}")
                return False
            
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces detected for {name}, using first one")
            
            # Encode the first face
            rgb_image = image[:, :, ::-1]
            face_encoding = face_recognition.face_encodings(
                rgb_image, 
                known_face_locations=[face_locations[0]],
                num_jitters=Config.NUM_JITTERS
            )[0]
            
            # Add to database
            self.encoder. add_encoding(face_encoding, name)
            self.encoder.save_encodings()
            
            logger.info(f"Successfully added {name} to database")
            return True
            
        except Exception as e:
            logger. error(f"Error adding person {name}: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """Get statistics about the recognition system."""
        encodings_dict = self.encoder. get_encodings_dict()
        
        return {
            'total_people': len(encodings_dict),
            'total_encodings': len(self.encoder. known_encodings),
            'average_encodings_per_person': (
                len(self.encoder.known_encodings) / len(encodings_dict) 
                if len(encodings_dict) > 0 else 0
            ),
            'people':  list(encodings_dict.keys())
        }