import tensorflow as tf
import numpy as np

# Check if TensorFlow can access the GPU
print("Is GPU available:", tf.test.is_gpu_available())

# List all available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f" - {gpu.name}")
else:
    print("No GPUs found.")

# Simple test model for training on GPU
def test_gpu_training_and_prediction():
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Generate random data to simulate training
    x_train = tf.random.normal([1000, 784])  # 1000 samples, 784 features (like MNIST)
    y_train = tf.random.uniform([1000], minval=0, maxval=10, dtype=tf.int32)  # 1000 labels (0-9)

    # Train the model for a few epochs
    model.fit(x_train, y_train, epochs=3, batch_size=32)

    # Generate random test data for prediction
    x_test = tf.random.normal([5, 784])  # 5 test samples

    # Perform predictions
    predictions = model.predict(x_test)
    print("\nPredictions on test data:")
    print(predictions)

# Run the test if GPU is available
if tf.test.is_gpu_available():
    test_gpu_training_and_prediction()
else:
    print("GPU is not available for training.")
