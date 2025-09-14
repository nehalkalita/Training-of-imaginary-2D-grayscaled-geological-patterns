import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import os
from PIL import Image

# Configuration (matching your original code)
IMG_SIZE = 25  # Image resolution (25x25)
DATA_DIRS = [
    "class1",
    "class2",
    "class3"
]
CLASS_NAMES = ['Artefact', 'Peak', 'Flat']

# Function to load and preprocess a single image
def load_image(file_path, img_size=IMG_SIZE):
    try:
        img = Image.open(file_path).convert('L')  # Convert to grayscale
        #img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0.0, 1.0]
        img_array = img_array[..., np.newaxis]  # Add channel dimension
        return img_array
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# Function to get a sample image from the dataset
def get_sample_image(data_dirs):
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_arry, locations = [], []
    for dir_path in data_dirs:
        image_arry.append([])
        locations.append([])
        for file_name in os.listdir(dir_path):
            if file_name.lower().endswith(supported_extensions):
                file_path = os.path.join(dir_path, file_name)
                img_array = load_image(file_path)
                if img_array is not None:
                    #return img_array, file_path
                    image_arry[-1].append(img_array)
                    locations[-1].append(file_path)
    return image_arry[1][2], locations[1][2]
    raise FileNotFoundError("No valid images found in the provided directories.")

# Function to visualize feature maps
def visualize_feature_maps(model, sample_image, layer_name='conv2d'):
    # Create a model that outputs the activations of the specified layer
    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    if not layer_outputs:
        raise ValueError(f"Layer {layer_name} not found in the model.")
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    # Get feature maps for the sample image
    sample_image_batch = np.expand_dims(sample_image, axis=0)  # Add batch dimension
    feature_maps = activation_model.predict(sample_image_batch)[0]  # Shape: (25, 25, 32)
    
    # Plot the original image and feature maps
    num_filters = feature_maps.shape[-1]  # 32 filters
    rows = 4
    cols = 8  # For 32 feature maps
    plt.figure(figsize=(cols * 2, rows * 2 + 1))
    
    # Plot original image
    plt.subplot(rows + 1, cols, 1)
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot each feature map
    for i in range(num_filters):
        plt.subplot(rows + 1, cols, i + cols + 1)  # Start after first row
        feature_map = feature_maps[:, :, i]
        plt.imshow(feature_map, cmap='gray')  # Use 'viridis' for better contrast. Other option is 'hot'
        plt.title(f'Filter {i+1}')
        plt.axis('off')
        
        # Calculate and display mean border vs. center activations
        border_pixels = np.concatenate([
            feature_map[0, :],  # Top row
            feature_map[-1, :],  # Bottom row
            feature_map[:, 0],  # Left column
            feature_map[:, -1]  # Right column
        ])
        center_pixels = feature_map[1:-1, 1:-1]  # Inner pixels
        mean_border = np.mean(border_pixels) if border_pixels.size > 0 else 0
        mean_center = np.mean(center_pixels) if center_pixels.size > 0 else 0
        plt.text(0, -1, f'B: {mean_border:.2f}\nC: {mean_center:.2f}', color='white', fontsize=6)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the trained model
    try:
        model = load_model('cnn_model_b.keras')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get a sample image
    try:
        sample_image, image_path = get_sample_image(DATA_DIRS)
        print(f"Visualizing feature maps for image: {image_path}")
    except FileNotFoundError as e:
        print(e)
        return
    
    # Visualize feature maps from the 'conv2d' layer
    visualize_feature_maps(model, sample_image, layer_name='conv2d')

if __name__ == '__main__':
    main()