import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    FACES_DIR = DATA_DIR / "faces"
    ENCODINGS_DIR = DATA_DIR / "encodings"
    MODELS_DIR = BASE_DIR / "models"
    
    # Create directories if they don't exist
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    ENCODINGS_DIR. mkdir(parents=True, exist_ok=True)
    
    # Face detection settings
    DETECTION_METHOD = "hog"  # Options: 'hog', 'cnn'
    MIN_DETECTION_CONFIDENCE = 0.5
    FACE_DETECTION_MODEL = "haar"  # Options: 'haar', 'dnn', 'mtcnn'
    
    # Face recognition settings
    RECOGNITION_THRESHOLD = 0.6
    FACE_SIZE = (160, 160)
    NUM_JITTERS = 1  # Higher = more accurate but slower
    
    # Video settings
    VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
    FRAME_WIDTH = 480  # Reduced from 640 for better performance
    FRAME_HEIGHT = 360  # Reduced from 480 for better performance
    FPS = 30
    FRAME_SKIP = 5  # Process every Nth frame (higher = faster but less responsive)
    
    # Display settings
    DISPLAY_FPS = True
    DISPLAY_CONFIDENCE = True
    BOX_COLOR = (0, 255, 0)
    TEXT_COLOR = (255, 255, 255)
    BOX_THICKNESS = 2
    
    # Training settings
    TRAIN_TEST_SPLIT = 0.2
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///face_recognition.db")
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = BASE_DIR / "logs" / "face_recognition.log"
    
    # Performance
    USE_GPU = True
    NUM_WORKERS = 4
    
    # Security
    MAX_FACE_DISTANCE = 0.6
    ANTI_SPOOFING_ENABLED = True
    
    # Web API
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"