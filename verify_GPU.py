import tensorflow as tf

def check_gpu():
    # Enable logging of device placement (optional, useful for debugging)
    tf.debugging.set_log_device_placement(True)
    
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"GPUs Available: {len(physical_devices)}")
    for gpu in physical_devices:
        print(f"  - {gpu.name}")
    
    # Check if TensorFlow is built with CUDA
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    # Check if GPU is available
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

if __name__ == "__main__":
    # Configure TensorFlow to allow memory growth on GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for each GPU to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    check_gpu()
