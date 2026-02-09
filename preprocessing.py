import tensorflow as tf
from tensorflow import keras
# import matplotlib as plt

print("SYSTEM/PACKAGE INFORMATION:")
print("TensorFlow version: ", tf.__version__)
print("keras version", keras.__version__)
print("GPU built with TensorFlow: ", tf.test.is_built_with_cuda())
print("Can access GPU: ", tf.config.experimental.list_physical_devices('GPU'))
print("TROUBLESHOOT DONE")

'''TRAIN-VALIDATE-TEST SPLIT'''
model_labels = ['L', 'F'] # L is leaf, F is flower
folders = ["Training", "Validation", "Testing"]
plant_categories = ["AAU", "ACO", "AMA", "ARH", "BPU", "CFI", "CJA", "CRA", "DRE", "IBI", "IPA", "LLE", "LPU", "MDI",
"MPU", "MQU", "PDU", "PIN", "PSA", "PVU", "SAL", "SOB", "SOC", "SSA", "TIN"]

directory = None #insert local storage path to folder of images

# Resizing and converting images to png


# Image Augmentation (for necessary classes)
preprocessing_pipeline = tf.keras.layers.Pipeline([
    tf.keras.layers.RandomRotation(20),
    tf.keras.layers.RandomFlip(), # horizontal_and_vertical by default
    tf.keras.layers.RandomCrop(30, 30),
    tf.keras.layers.RandomBrightness((0.8, 1.2))
])

# Train-Validate-Test Split
train_val_dataset = tf.keras.utils.image_dataset_from_directory(directory, labels = 'inferred', label_mode = 'categorical',
                                                                image_size = (224, 224), shuffle = False, validation_split = 0.3, subset = 'both')
validate_ds, test_ds = tf.keras.utils.split_dataset(train_val_dataset, left_size=0.5, right_size=None, shuffle=False, seed=1)

print("DATASETS:")
print(train_val_dataset)
print(validate_ds)
print(test_ds)
print()