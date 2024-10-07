import tensorflow_hub as hub
import tensorflow as tf

# Load the FaceNet model from TensorFlow Hub
def load_facenet_model():
    model_url = "https://tfhub.dev/danielhnyk/resnet50_facenet/1"
    model = hub.load(model_url)
    return model

# Function to generate embeddings from an image
def get_embedding(model, image):
    # Resize image to 160x160 as required by FaceNet
    image = tf.image.resize(image, (160, 160))
    
    # Convert image to float32 type and normalize pixel values
    image = tf.cast(image, tf.float32) / 255.0

    # Add batch dimension and pass image through model to get the embedding
    image = tf.expand_dims(image, axis=0)
    embedding = model(image)
    
    # Return the embedding as a numpy array
    return embedding[0].numpy()
