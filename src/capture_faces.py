import cv2
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config
import time

def capture_faces_for_person(name:  str, num_images: int = 10):
    """
    Capture multiple photos of a person for training. 
    
    Args:
        name: Person's name
        num_images: Number of images to capture
    """
    # Create directory for this person
    person_dir = Config.FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📸 Capturing {num_images} photos for:  {name}")
    print(f"{'='*60}")
    print("\nInstructions:")
    print("  - Look at the camera")
    print("  - Change your angle slightly between photos")
    print("  - Try different expressions")
    print("  - Ensure good lighting")
    print(f"\nPhotos will be saved to: {person_dir}")
    print("\nPress SPACE to capture, Q to quit\n")
    
    # Open webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("❌ Error: Could not open webcam")
        return False
    
    # Load face detector for preview
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    captured_count = 0
    
    print("📷 Webcam opened.  Position yourself and press SPACE to capture.")
    
    while captured_count < num_images: 
        ret, frame = video_capture.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Detect faces for preview
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Face Detected",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Display counter and instructions
        cv2.putText(
            frame,
            f"Captured: {captured_count}/{num_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        cv2.putText(
            frame,
            "SPACE:  Capture | Q: Quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Show frame
        cv2.imshow(f'Capturing photos for {name}', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar
            if len(faces) == 0:
                print("⚠️  No face detected!  Please position yourself properly.")
                continue
            
            if len(faces) > 1:
                print("⚠️  Multiple faces detected! Only one person should be visible.")
                continue
            
            # Save the image
            timestamp = int(time.time() * 1000)
            filename = person_dir / f"{name}_{captured_count+1}_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            
            captured_count += 1
            print(f"✅ Captured {captured_count}/{num_images}:  {filename. name}")
            
            # Brief pause for feedback
            time.sleep(0.5)
        
        elif key == ord('q'):
            print("\n❌ Capture cancelled by user")
            break
    
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    
    if captured_count == num_images:
        print(f"\n{'='*60}")
        print(f"✅ Successfully captured {captured_count} photos for {name}!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"  1. Capture photos for other people (if needed)")
        print(f"  2. Run:  python src/train.py")
        print(f"  3. Run: python src/recognize. py")
        return True
    else:
        print(f"\n⚠️  Only captured {captured_count}/{num_images} photos")
        return False

def main():
    """Main function for interactive face capture."""
    print("="*60)
    print("Face Recognition System - Photo Capture Tool")
    print("="*60)
    
    # Get person's name
    name = input("\nEnter person's name: ").strip()
    
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    # Get number of images
    while True:
        try:
            num_images = input("How many photos to capture? (recommended: 10-20, press Enter for 10): ").strip()
            if not num_images:
                num_images = 10
                break
            num_images = int(num_images)
            if 5 <= num_images <= 50:
                break
            else:
                print("⚠️  Please enter a number between 5 and 50")
        except ValueError:
            print("⚠️  Please enter a valid number")
    
    # Start capture
    success = capture_faces_for_person(name, num_images)
    
    if success:
        # Ask if user wants to add another person
        another = input("\nCapture photos for another person? (y/n): ").strip().lower()
        if another == 'y':
            main()

if __name__ == "__main__":
    main()