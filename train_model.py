import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Pool, cpu_count, freeze_support, RawArray, Manager
import os
from PIL import Image
import ctypes

# Configuration
IMG_SIZE = 25
NUM_CLASSES = 3
BATCH_SIZE = 4
EPOCHS = 50
NUM_WORKERS = cpu_count()
DATA_DIRS = [
    "class1",
    "class2",
    "class3"
]
CLASS_NAMES = ['Artefact', 'Peak', 'Flat']

# Disable GPU and oneDNN optimizations
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress oneDNN warning

# Global shared memory variables
X_buffer = None
Y_buffer = None
X_shape = None
Y_shape = None

# Function to initialize worker processes
def init_worker(x_buf, y_buf, x_shp, y_shp):
    global X_buffer, Y_buffer, X_shape, Y_shape
    X_buffer = x_buf
    Y_buffer = y_buf
    X_shape = x_shp
    Y_shape = y_shp

# Function to load and preprocess a single image
def load_image(file_path, img_size=IMG_SIZE):
    try:
        img = Image.open(file_path).convert('L')
        img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# Worker function for multiprocessing dataset creation
def process_images(args):
    dir_path, class_name = args
    images = []
    labels = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    for file_name in os.listdir(dir_path):
        if file_name.lower().endswith(supported_extensions):
            file_path = os.path.join(dir_path, file_name)
            img_array = load_image(file_path)
            if img_array is not None:
                images.append(img_array)
                labels.append(class_name)
    return images, labels

# Generate training dataset
def create_dataset():
    pool = Pool(processes=NUM_WORKERS)
    args = [(DATA_DIRS[i], CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
    results = pool.map(process_images, args)
    pool.close()
    pool.join()
    
    images = []
    labels = []
    for img_list, lbl_list in results:
        images.extend(img_list)
        labels.extend(lbl_list)
    
    images = np.array(images)
    labels = np.array(labels)
    
    label_to_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    numerical_labels = np.array([label_to_index[label] for label in labels])
    
    images = images[..., np.newaxis]
    indices = np.random.permutation(len(images))
    images = images[indices]
    numerical_labels = numerical_labels[indices]
    labels_one_hot = tf.keras.utils.to_categorical(numerical_labels, NUM_CLASSES)
    
    class_counts = np.bincount(numerical_labels)
    class_distribution = {CLASS_NAMES[i]: class_counts[i] for i in range(len(class_counts))}
    print(f"Training set class distribution: {class_distribution}")
    if len(class_counts) < NUM_CLASSES:
        raise ValueError("Not all classes are represented in the training set.")
    
    return images, labels_one_hot, label_to_index

# Define custom CNN model
def build_model(img_size, num_classes):
    inputs = Input(shape=(img_size, img_size, 1))
    conv1 = Conv2D(32, (2, 2), activation='relu', padding='same')(inputs)
    flat = Flatten()(conv1)
    dense1 = Dense(128, activation='relu')(flat)
    dropout = Dropout(0.1)(dense1)
    outputs = Dense(num_classes, activation='softmax')(dropout)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.0001, weight_decay=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Data augmentation
def create_data_generator():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.0,
        height_shift_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# Initialize shared memory
def init_shared_memory(x_train, y_train):
    global X_buffer, Y_buffer, X_shape, Y_shape
    X_shape = x_train.shape
    Y_shape = y_train.shape
    X_buffer = RawArray(ctypes.c_float, int(np.prod(X_shape)))
    Y_buffer = RawArray(ctypes.c_float, int(np.prod(Y_shape)))
    
    x_shared = np.frombuffer(X_buffer, dtype=np.float32).reshape(X_shape)
    y_shared = np.frombuffer(Y_buffer, dtype=np.float32).reshape(Y_shape)
    np.copyto(x_shared, x_train)
    np.copyto(y_shared, y_train)
    
    # Return shapes for use in worker initialization
    return X_shape, Y_shape

# Worker function for training
def train_model_worker(args):
    worker_id, num_epochs = args
    print(f"Worker {worker_id} training for {num_epochs} epochs...")
    
    # Access global shared memory
    if X_buffer is None or Y_buffer is None:
        raise ValueError(f"Worker {worker_id}: Shared memory not initialized")
    
    x_train = np.frombuffer(X_buffer, dtype=np.float32).reshape(X_shape)
    y_train = np.frombuffer(Y_buffer, dtype=np.float32).reshape(Y_shape)
    
    # Initialize TensorFlow in worker
    tf.keras.backend.clear_session()
    
    # Build and train model
    model = build_model(IMG_SIZE, NUM_CLASSES)
    datagen = create_data_generator()
    model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=num_epochs,
        verbose=1
    )
    
    return model.get_weights()

# Average model weights
def average_model_weights(weights_list):
    avg_weights = []
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

# Main training function
def main():
    for dir_path in DATA_DIRS:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist")
    
    if len(CLASS_NAMES) != NUM_CLASSES:
        raise ValueError(f"Number of class names ({len(CLASS_NAMES)}) does not match number of classes ({NUM_CLASSES})")
    
    print("Loading and preprocessing images...")
    x_train, y_train, label_to_index = create_dataset()
    
    if len(x_train) < 15:
        print(f"Warning: Only {len(x_train)} training images loaded. Consider adding more images or stronger augmentation.")
    
    # Initialize shared memory
    x_shape, y_shape = init_shared_memory(x_train, y_train)
    
    # Divide epochs among workers
    base_epochs = EPOCHS // NUM_WORKERS
    extra_epochs = EPOCHS % NUM_WORKERS
    epoch_assignments = [(i, base_epochs + (1 if i < extra_epochs else 0)) for i in range(NUM_WORKERS)]
    
    print(f"Starting parallel training with {NUM_WORKERS} workers...")
    # Initialize pool with shared memory
    pool = Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(X_buffer, Y_buffer, x_shape, y_shape))
    weights_list = pool.map(train_model_worker, epoch_assignments)
    pool.close()
    pool.join()
    
    print("Combining model weights...")
    avg_weights = average_model_weights(weights_list)
    
    final_model = build_model(IMG_SIZE, NUM_CLASSES)
    final_model.set_weights(avg_weights)
    
    final_model.save("cnn_model_b.keras")
    np.save("class_names.npy", CLASS_NAMES)
    np.save("label_to_index.npy", label_to_index)
    print("Model saved as cnn_model_b.keras")
    print("Class names and label mapping saved as class_names.npy and label_to_index.npy")

if __name__ == "__main__":
    freeze_support()
    main()