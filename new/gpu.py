import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.set_logical_device_configuration(
                device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
            )
        print("TensorFlow is using the GPU")
    except Exception as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found, using CPU")
