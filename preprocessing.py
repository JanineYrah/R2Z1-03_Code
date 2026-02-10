import tensorflow as tf
from tensorflow import keras
import os
import PIL
from PIL import Image
# import matplotlib as plt

print("SYSTEM/PACKAGE INFORMATION:")
print("TensorFlow version: ", tf.__version__)
print("keras version", keras.__version__)
print("GPU built with TensorFlow: ", tf.test.is_built_with_cuda())
print("Can access GPU: ", tf.config.experimental.list_physical_devices('GPU'))
print("TROUBLESHOOT DONE")

# TRAIN-VALIDATE-TEST SPLIT
model_labels = ['L', 'F'] # L is leaf, F is flower
plant_categories = ["AAU", "ACO", "AMA", "ARH", "BPU", "CFI", "CJA", "CRA", "DRE", "IBI", "IPA", "LLE", "LPU", "MDI",
"MPU", "MQU", "PDU", "PIN", "PSA", "PVU", "SAL", "SOB", "SOC", "SSA", "TIN"]
to_augment = ["AMA_F", "ARH_L", "PIN_F"]
directory = '/home/r2z103/dataset'

'''
# Resizing and converting images to png
for dirpath, dirnames, filenames in os.walk(directory):
    for folder in dirnames:
        print(f"Current plant category: {folder}")
        images = os.listdir(f"/home/r2z103/dataset/{folder}")
    
        for image_filename in images:
            image_filename_no_extension = image_filename[:-5] if image_filename.endswith(".jpeg") else image_filename[:-4]
            plant_image = Image.open(f"/home/r2z103/dataset/{folder}/{image_filename}") # loads image in PIL format
            resized_image = plant_image.resize((224, 224))
            resized_image.save(f"/home/r2z103/res_dataset/{folder}/{image_filename_no_extension}.png")
            resized_image.close()
            print(f"{image_filename_no_extension} has been edited.")
'''
'''
# Image Augmentation (for necessary classes)
augmentation_pipeline = tf.keras.Sequential([
    tf.keras.layers.RandomRotation([-0.20, 0.20]),
    tf.keras.layers.RandomFlip(), # horizontal_and_vertical by default
    tf.keras.layers.RandomCrop(30, 30),
    tf.keras.layers.RandomBrightness((0.8, 1.2))
])
'''

leaf_directory = 'home/r2z103/res_dataset/leaf' # all png and resized to 224x224
flower_directory = 'home/r2z103/res_dataset/flower' # all png and resized to 224x224

# Train-Validate-Test Split
dataset = tf.keras.preprocessing.image_dataset_from_directory(directory, shuffle = False)
# train_ds, validate_test_ds = tf.keras.utils.split_dataset(dataset, left_size=0.7, right_size=0.3, shuffle=False, seed=1)
# validate_ds, test_ds = tf.keras.utils.split_dataset(validate_test_ds, left_size=0.5, right_size=0.5, shuffle=False, seed=1)

# validate_dataset, test_dataset = tf.keras.utils.split_dataset(val_test_ds, left_size=0.5, right_size=None, shuffle=False, seed=1)

print("DATASETS:")
print(dataset)