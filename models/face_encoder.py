import numpy as np
import face_recognition
from typing import List, Dict
import pickle
from pathlib import Path
from config.config import Config
from loguru import logger

class FaceEncoder:
    """
    Encode faces into 128-dimensional embeddings.
    """
    
    def __init__(self):
        self.encodings_path = Config.ENCODINGS_DIR / "face_encodings.pkl"
        self.known_encodings:  List[np.ndarray] = []
        self.known_names: List[str] = []
        logger.info("Face encoder initialized")
    
    def encode_face(self, image: np.ndarray, 
                   face_location: tuple) -> np.ndarray:
        """
        Generate 128-dimensional encoding for a face.
        
        Args:
            image: Input image (RGB format)
            face_location:  Face bounding box (top, right, bottom, left)
            
        Returns:
            128-dimensional face encoding
        """
        rgb_image = image if image.shape[2] == 3 else image[:, :, : :-1]
        
        encodings = face_recognition.face_encodings(
            rgb_image, 
            known_face_locations=[face_location],
            num_jitters=Config.NUM_JITTERS
        )
        
        if len(encodings) > 0:
            return encodings[0]
        return None
    
    def encode_faces_in_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Encode all faces found in an image.
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            List of face encodings
        """
        rgb_image = image if image.shape[2] == 3 else image[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_image)
        encodings = face_recognition.face_encodings(
            rgb_image, 
            known_face_locations=face_locations,
            num_jitters=Config.NUM_JITTERS
        )
        
        return encodings
    
    def load_encodings(self) -> bool:
        """
        Load pre-computed face encodings from disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.encodings_path.exists():
            logger.warning("No encodings file found")
            return False
        
        try:
            with open(self.encodings_path, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self. known_names = data['names']
            
            logger.info(f"Loaded {len(self.known_encodings)} face encodings")
            return True
        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
            return False
    
    def save_encodings(self) -> bool:
        """
        Save face encodings to disk. 
        
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                'encodings':  self.known_encodings,
                'names': self.known_names
            }
            
            with open(self.encodings_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(self.known_encodings)} face encodings")
            return True
        except Exception as e: 
            logger.error(f"Error saving encodings: {e}")
            return False
    
    def add_encoding(self, encoding: np.ndarray, name: str):
        """Add a new face encoding to the database."""
        self.known_encodings.append(encoding)
        self.known_names.append(name)
    
    def get_encodings_dict(self) -> Dict[str, List[np.ndarray]]: 
        """
        Get encodings organized by name.
        
        Returns:
            Dictionary mapping names to their encodings
        """
        encodings_dict = {}
        for encoding, name in zip(self.known_encodings, self.known_names):
            if name not in encodings_dict: 
                encodings_dict[name] = []
            encodings_dict[name].append(encoding)
        
        return encodings_dict