import cv2
import numpy as np
import tensorflow as tf
from facenet_model import load_facenet_model, get_embedding

# Load FaceNet model
facenet_model = load_facenet_model()

# Function to read and preprocess an image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Detect face using OpenCV
def detect_face(image):
    # Load the pre-trained face detection model (Haar Cascades)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert image to grayscale for detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return None, None
    
    # Extract the bounding box of the first face detected
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    
    return face, (x, y, w, h)

# Example: Load an image, detect face, and get embedding
image_path = "images/known_person1.jpg"
image = load_image(image_path)
face, bbox = detect_face(image)

if face is not None:
    embedding = get_embedding(facenet_model, face)
    print("Face Embedding:", embedding)
else:
    print("No face detected.")


# Calculate Euclidean distance between two embeddings
def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# Define a threshold for determining a match
THRESHOLD = 0.5

def is_match(embedding1, embedding2, threshold=THRESHOLD):
    distance = calculate_distance(embedding1, embedding2)
    return distance < threshold

# Example of comparing known and unknown faces
known_image_path = "images/known_person1.jpg"
unknown_image_path = "images/unknown_person.jpg"

# Get embeddings for known and unknown faces
known_image = load_image(known_image_path)
known_face, _ = detect_face(known_image)
known_embedding = get_embedding(facenet_model, known_face)

unknown_image = load_image(unknown_image_path)
unknown_face, _ = detect_face(unknown_image)
unknown_embedding = get_embedding(facenet_model, unknown_face)

# Check if the faces match
if is_match(known_embedding, unknown_embedding):
    print("Faces match!")
else:
    print("Faces do not match.")
