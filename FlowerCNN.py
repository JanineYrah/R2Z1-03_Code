# This will display "Hello, World!" in the terminal
print("Hello, World!")

# This will display your system version in the terminal
import sys
print(sys.version)

import tensorflow as tf
from tensorflow import keras

print("tf version", tf.__version__)
print("keras version", keras.__version__)

print("TensorFlow version: ", tf.__version__)
print("GPU built with TensorFlow: ", tf.test.is_built_with_cuda())
print("Can access GPU: ", tf.config.experimental.list_physical_devices('GPU'))
print('67 poggers nyancat')