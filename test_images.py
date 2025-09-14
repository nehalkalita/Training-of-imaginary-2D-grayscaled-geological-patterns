import numpy as np
import tensorflow as tf
from multiprocessing import Pool, cpu_count, freeze_support
import os
from PIL import Image

# Configuration
IMG_SIZE = 25  # Must match training configuration
NUM_WORKERS = cpu_count()
TEST_DIR = "test_images"  # Update with your test directory

# Disable GPU to ensure CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Function to load and preprocess a single image
def load_image(file_path, img_size=IMG_SIZE):
    try:
        img = Image.open(file_path).convert('L')  # Convert to grayscale
        img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0.0, 1.0]
        return img_array
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# Worker function for preprocessing images
def preprocess_image(file_path):
    img_array = load_image(file_path)
    if img_array is not None:
        return img_array[..., np.newaxis]  # Add channel dimension
    return None

# Load test images from directory
def load_test_images(test_dir):
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    file_paths = [
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if f.lower().endswith(supported_extensions)
    ]
    return file_paths

# Main testing function
def main():
    # Validate test directory
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory {TEST_DIR} does not exist")
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model("cnn_model_b.keras")
    
    # Load class names
    print("Loading class names...")
    if not os.path.exists("class_names.npy"):
        raise FileNotFoundError("class_names.npy not found. Ensure it was saved during training.")
    class_names = np.load("class_names.npy", allow_pickle=True)
    
    # Get test image paths
    print("Collecting test image paths...")
    test_files = load_test_images(TEST_DIR)
    
    if not test_files:
        raise ValueError("No valid test images found")
    
    # Preprocess images using multiprocessing
    print(f"Preprocessing {len(test_files)} test images...")
    pool = Pool(processes=NUM_WORKERS)
    test_images = pool.map(preprocess_image, test_files)
    pool.close()
    pool.join()
    
    # Filter out None values (failed loads)
    valid_files = []
    valid_images = []
    for file_path, img in zip(test_files, test_images):
        if img is not None:
            valid_files.append(file_path)
            valid_images.append(img)
    
    if not valid_images:
        raise ValueError("No valid test images could be processed")
    
    test_images = np.array(valid_images)
    
    # Predict
    print("Making predictions...")
    predictions = model.predict(test_images)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_classes = [class_names[idx] for idx in predicted_indices]
    
    # Display results
    for i, (file_path, pred_class) in enumerate(zip(valid_files, predicted_classes)):
        print(f"Image {i+1} ({os.path.basename(file_path)}): Predicted class = {pred_class}")

if __name__ == "__main__":
    freeze_support()
    main()