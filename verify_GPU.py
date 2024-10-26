import tensorflow as tf

def check_gpu():
    print("TensorFlow version:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

if __name__ == "__main__":
    check_gpu()
