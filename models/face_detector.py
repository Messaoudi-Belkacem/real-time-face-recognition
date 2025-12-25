import cv2
import numpy as np
from typing import List, Tuple, Optional
import face_recognition
from config. config import Config
from loguru import logger

class FaceDetector:
    """
    Advanced face detection using multiple methods. 
    Supports Haar Cascades, HOG, CNN, and DNN-based detection.
    
    This class provides a unified interface for face detection using different
    algorithms, each with its own trade-offs:
    
    - 'haar': Fast, works on CPU, less accurate, good for simple scenarios
    - 'hog': Moderate speed, CPU-friendly, good accuracy for frontal faces
    - 'cnn': Slow on CPU but very accurate, best with GPU
    - 'dnn': Fast and accurate, works well on CPU, uses deep learning
    
    Attributes:
        method (str): Detection method to use ('haar', 'hog', 'cnn', or 'dnn')
        confidence_threshold (float): Minimum confidence for DNN detections
        face_cascade (cv2.CascadeClassifier): Haar cascade classifier (if using 'haar')
        net (cv2.dnn_Net): DNN model (if using 'dnn')
    
    Example:
        >>> detector = FaceDetector(method='hog')
        >>> image = cv2.imread('photo.jpg')
        >>> faces = detector.detect_faces(image)
        >>> print(f"Found {len(faces)} faces")
    """
    
    def __init__(self, method: str = Config.DETECTION_METHOD):
        """
        Initialize the face detector with the specified method.
        
        Args:
            method (str): Detection method to use. Options:
                - 'haar': Haar Cascade Classifier (fast, CPU-friendly)
                - 'hog': Histogram of Oriented Gradients (balanced)
                - 'cnn': Convolutional Neural Network (accurate, needs GPU)
                - 'dnn': Deep Neural Network with Caffe model (fast & accurate)
        
        Raises:
            FileNotFoundError: If DNN model files are not found
            ValueError: If invalid detection method is specified
        
        Note:
            For 'dnn' method, ensure deploy.prototxt and .caffemodel files
            are present in the models/ directory.
        """
        self.method = method
        self.confidence_threshold = Config.MIN_DETECTION_CONFIDENCE
        
        if method == 'haar':
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Initialized Haar Cascade detector")
            
        elif method == 'dnn':
            # Load DNN model for face detection
            prototxt = "models/deploy.prototxt"
            model = "models/res10_300x300_ssd_iter_140000.caffemodel"
            self. net = cv2.dnn.readNetFromCaffe(prototxt, model)
            logger. info("Initialized DNN detector")
            
        logger.info(f"Face detector initialized with method: {method}")
    
    def detect_faces(self, image: np.ndarray, 
                     scale_factor: float = 1.0) -> List[Tuple[int, int, int, int]]: 
        """
        Detect faces in an image using the configured detection method.
        
        This is the main interface for face detection. It delegates to the
        appropriate detection method based on the instance configuration.
        
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV default).
                               Should be a valid numpy array with shape (H, W, 3).
            scale_factor (float, optional): Scale factor for faster processing.
                                          Values < 1.0 will resize image smaller
                                          for faster detection. Detected faces are
                                          scaled back to original coordinates.
                                          Default: 1.0 (no scaling)
        
        Returns:
            List[Tuple[int, int, int, int]]: List of face bounding boxes.
                Each box is a tuple of (top, right, bottom, left) coordinates
                in pixels, representing the face location in the original image.
                Returns empty list if no faces detected.
        
        Raises:
            ValueError: If an unknown detection method is configured
        
        Example:
            >>> detector = FaceDetector(method='hog')
            >>> image = cv2.imread('group_photo.jpg')
            >>> faces = detector.detect_faces(image, scale_factor=0.5)
            >>> for (top, right, bottom, left) in faces:
            ...     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        Note:
            - Smaller scale_factor = faster but may miss small faces
            - DNN method ignores scale_factor and uses fixed 300x300 input
            - Face coordinates are always in original image space
        """
        if self.method == 'haar': 
            return self._detect_haar(image, scale_factor)
        elif self.method == 'hog':
            return self._detect_hog(image, scale_factor)
        elif self.method == 'cnn':
            return self._detect_cnn(image, scale_factor)
        elif self.method == 'dnn':
            return self._detect_dnn(image, scale_factor)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _detect_haar(self, image: np. ndarray, 
                     scale_factor: float) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar Cascade Classifier.
        
        This is a classical computer vision approach based on Haar-like features
        and AdaBoost. It's fast and works well on CPU but can produce false positives
        and is sensitive to face orientation.
        
        Args:
            image (np.ndarray): Input image in BGR format
            scale_factor (float): Scale factor for resizing image before detection
        
        Returns:
            List[Tuple[int, int, int, int]]: Face bounding boxes in format
                                            (top, right, bottom, left)
        
        Implementation Details:
            - Converts image to grayscale (Haar works on intensity)
            - Uses scaleFactor=1.1 for multi-scale detection
            - minNeighbors=5 to reduce false positives
            - minSize=(30, 30) to ignore very small detections
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if scale_factor != 1.0:
            small_image = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
        else:
            small_image = gray
        
        faces = self.face_cascade.detectMultiScale(
            small_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to (top, right, bottom, left) format
        face_locations = []
        for (x, y, w, h) in faces:
            if scale_factor != 1.0:
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)
            face_locations.append((y, x + w, y + h, x))
        
        return face_locations
    
    def _detect_hog(self, image: np.ndarray, 
                    scale_factor: float) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using HOG (Histogram of Oriented Gradients).
        
        HOG is a feature descriptor that works by extracting gradient orientation
        histograms. It provides a good balance between speed and accuracy, and
        works well for frontal faces in controlled lighting.
        
        Args:
            image (np.ndarray): Input image in BGR format
            scale_factor (float): Scale factor for resizing image before detection.
                                 Recommended: 0.5 for faster processing
        
        Returns:
            List[Tuple[int, int, int, int]]: Face bounding boxes in format
                                            (top, right, bottom, left)
        
        Note:
            - Uses face_recognition library's implementation (dlib-based)
            - number_of_times_to_upsample controls detection of smaller faces
            - More upsampling = slower but detects smaller faces
            - Best for frontal or near-frontal faces
            - RGB conversion required (dlib expects RGB)
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if scale_factor != 1.0:
            small_image = cv2.resize(rgb_image, None, fx=scale_factor, fy=scale_factor)
            face_locations = face_recognition.face_locations(
                small_image, 
                model="hog",
                number_of_times_to_upsample=Config.NUM_JITTERS
            )
            # Scale back to original size
            face_locations = [
                (int(top / scale_factor), int(right / scale_factor),
                 int(bottom / scale_factor), int(left / scale_factor))
                for (top, right, bottom, left) in face_locations
            ]
        else:
            face_locations = face_recognition.face_locations(
                rgb_image, 
                model="hog",
                number_of_times_to_upsample=Config.NUM_JITTERS
            )
        
        return face_locations
    
    def _detect_cnn(self, image: np. ndarray, 
                    scale_factor: float) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using CNN (Convolutional Neural Network).
        
        This method uses a deep learning model for face detection, providing
        the highest accuracy among all methods. However, it's significantly
        slower on CPU and requires GPU (CUDA) for real-time performance.
        
        Args:
            image (np.ndarray): Input image in BGR format
            scale_factor (float): Scale factor for resizing image before detection.
                                 Highly recommended to use 0.25-0.5 for speed
        
        Returns:
            List[Tuple[int, int, int, int]]: Face bounding boxes in format
                                            (top, right, bottom, left)
        
        Performance Notes:
            - CPU: ~2-5 seconds per frame (very slow)
            - GPU (CUDA): ~30-60 FPS (real-time capable)
            - Use scale_factor < 1.0 for speed improvement
        
        Advantages:
            - Most accurate detection
            - Handles profile faces and occlusions better
            - Works across various lighting conditions
        
        Warning:
            Not recommended for real-time CPU applications without scaling.
            Consider 'hog' or 'dnn' methods for CPU-based detection.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if scale_factor != 1.0:
            small_image = cv2.resize(rgb_image, None, fx=scale_factor, fy=scale_factor)
            face_locations = face_recognition.face_locations(
                small_image, 
                model="cnn",
                number_of_times_to_upsample=0
            )
            # Scale back to original size
            face_locations = [
                (int(top / scale_factor), int(right / scale_factor),
                 int(bottom / scale_factor), int(left / scale_factor))
                for (top, right, bottom, left) in face_locations
            ]
        else: 
            face_locations = face_recognition.face_locations(
                rgb_image, 
                model="cnn",
                number_of_times_to_upsample=0
            )
        
        return face_locations
    
    def _detect_dnn(self, image: np.ndarray, 
                    scale_factor: float) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN-based detector (Caffe ResNet SSD).
        
        This method uses OpenCV's DNN module with a pre-trained ResNet-10
        Single Shot Detector model. It provides excellent accuracy and speed
        on CPU, making it ideal for production applications.
        
        Args:
            image (np.ndarray): Input image in BGR format
            scale_factor (float): Not used in this method (DNN uses fixed 300x300)
        
        Returns:
            List[Tuple[int, int, int, int]]: Face bounding boxes in format
                                            (top, right, bottom, left)
        
        Model Details:
            - Architecture: ResNet-10 backbone with SSD
            - Input size: 300x300 (fixed)
            - Mean subtraction: (104.0, 177.0, 123.0) - BGR
            - Confidence threshold: Configured in Config.MIN_DETECTION_CONFIDENCE
        
        Implementation:
            1. Resize image to 300x300 and create blob
            2. Forward pass through network
            3. Filter detections by confidence threshold
            4. Scale coordinates back to original image size
        
        Advantages:
            - Fast on CPU (~30-60 FPS on modern processors)
            - Good accuracy for frontal and near-frontal faces
            - Consistent performance across different hardware
            - No GPU required
        
        Note:
            Requires deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel
            files in the models/ directory.
        """
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self. net.setInput(blob)
        detections = self.net. forward()
        
        face_locations = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Convert to (top, right, bottom, left) format
                face_locations.append((startY, endX, endY, startX))
        
        return face_locations
    
    def detect_faces_with_landmarks(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Detect faces and extract facial landmarks in a single operation.
        
        This method detects faces and identifies key facial features (landmarks)
        such as eyes, nose, mouth, chin, and eyebrows. Useful for face alignment,
        emotion detection, and face preprocessing.
        
        Args:
            image (np.ndarray): Input image in BGR format
        
        Returns:
            Tuple[List, List]: A tuple containing:
                - face_locations: List of bounding boxes (top, right, bottom, left)
                - face_landmarks: List of dictionaries, each containing:
                    - 'chin': List of (x, y) points along the chin
                    - 'left_eyebrow': Left eyebrow points
                    - 'right_eyebrow': Right eyebrow points
                    - 'nose_bridge': Nose bridge points
                    - 'nose_tip': Nose tip points
                    - 'left_eye': Left eye points
                    - 'right_eye': Right eye points
                    - 'top_lip': Upper lip outline points
                    - 'bottom_lip': Lower lip outline points
        
        Example:
            >>> detector = FaceDetector()
            >>> image = cv2.imread('portrait.jpg')
            >>> locations, landmarks = detector.detect_faces_with_landmarks(image)
            >>> 
            >>> for face_landmarks in landmarks:
            ...     # Draw eye landmarks
            ...     for (x, y) in face_landmarks['left_eye']:
            ...         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        Note:
            - Uses face_recognition library (dlib 68-point model)
            - Returns 68 facial landmarks per face
            - More computationally expensive than detection alone
            - Landmarks are in (x, y) pixel coordinates
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image, model=self.method)
        face_landmarks = face_recognition.face_landmarks(rgb_image, face_locations)
        
        return face_locations, face_landmarks