import cv2
import os
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.face_encoder import FaceEncoder
from models.face_detector import FaceDetector
from config.config import Config
from loguru import logger
import face_recognition

def train_from_dataset(dataset_path: Path = Config.FACES_DIR) -> bool:
    """
    Train the face recognition system from a dataset of images.
    
    Dataset structure:
        faces/
            person1/
                image1.jpg
                image2.jpg
            person2/
                image1.jpg
                image2.jpg
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting training process...")
    
    detector = FaceDetector()
    encoder = FaceEncoder()
    
    # Clear existing encodings
    encoder.known_encodings = []
    encoder.known_names = []
    
    # Get all person directories
    person_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(person_dirs) == 0:
        logger.error(f"No person directories found in {dataset_path}")
        return False
    
    total_faces = 0
    failed_images = []
    
    # Process each person
    for person_dir in tqdm(person_dirs, desc="Processing people"):
        person_name = person_dir.name
        logger.info(f"Processing images for {person_name}")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(person_dir.glob(ext))
        
        if len(image_files) == 0:
            logger.warning(f"No images found for {person_name}")
            continue
        
        person_face_count = 0
        
        # Process each image
        for image_path in tqdm(image_files, desc=f"  {person_name}", leave=False):
            try:
                # Load image
                image = cv2.imread(str(image_path))
                
                if image is None:
                    logger.warning(f"Could not load image:  {image_path}")
                    failed_images.append(str(image_path))
                    continue
                
                # Convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(
                    rgb_image, 
                    model="hog"
                )
                
                if len(face_locations) == 0:
                    logger.warning(f"No face detected in {image_path}")
                    failed_images.append(str(image_path))
                    continue
                
                if len(face_locations) > 1:
                    logger.warning(f"Multiple faces in {image_path}, using first")
                
                # Encode face
                face_encodings = face_recognition.face_encodings(
                    rgb_image, 
                    known_face_locations=face_locations,
                    num_jitters=Config.NUM_JITTERS
                )
                
                if len(face_encodings) > 0:
                    encoder.add_encoding(face_encodings[0], person_name)
                    person_face_count += 1
                    total_faces += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                failed_images.append(str(image_path))
        
        logger.info(f"Encoded {person_face_count} faces for {person_name}")
    
    # Save encodings
    if total_faces > 0:
        encoder.save_encodings()
        logger.info(f"Training complete!  Total faces encoded: {total_faces}")
        
        if failed_images:
            logger. warning(f"Failed to process {len(failed_images)} images:")
            for img in failed_images[: 10]:  # Show first 10
                logger.warning(f"  - {img}")
        
        return True
    else: 
        logger.error("No faces were encoded!")
        return False

def main():
    """Main training function."""
    logger.add(Config.LOG_FILE, rotation="10 MB")
    
    print("=" * 60)
    print("Face Recognition System - Training")
    print("=" * 60)
    print(f"\nDataset path: {Config.FACES_DIR}")
    print(f"Detection method: {Config.DETECTION_METHOD}")
    print(f"Number of jitters: {Config.NUM_JITTERS}")
    print("\nMake sure your dataset is organized as:")
    print("  faces/")
    print("    person1/")
    print("      image1.jpg")
    print("      image2.jpg")
    print("    person2/")
    print("      image1.jpg")
    print("\n" + "=" * 60)
    
    input("\nPress Enter to start training...")
    
    success = train_from_dataset()
    
    if success:
        print("\n✓ Training completed successfully!")
        print(f"Encodings saved to: {Config. ENCODINGS_DIR / 'face_encodings.pkl'}")
    else:
        print("\n✗ Training failed!")
    
    return success

if __name__ == "__main__":
    main()