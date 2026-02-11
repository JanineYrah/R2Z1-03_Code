import tensorflow as tf
from tensorflow import keras
import os
import PIL
from PIL import Image
import matplotlib as plt
import numpy as np

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

leaf_directory = '/home/r2z103/res_dataset/Leaf' # all png and resized to 224x224
flower_directory = '/home/r2z103/res_dataset/Flower' # all png and resized to 224x224

# Train-Validate-Test Split
# set batch_size = 1 to see true number of images, as the cardinality would give the number of image batches (based on batch_size = 32)
print("DATASETS:")
print("LEAF")
leaf_train_ds, leaf_val_test_ds = tf.keras.utils.image_dataset_from_directory(leaf_directory, labels = "inferred", label_mode = "int", validation_split = 0.3,
                                                                                      subset = "both", color_mode = "rgb", shuffle = True, seed = 1, batch_size = 1)
leaf_val_batches = tf.data.experimental.cardinality(leaf_val_test_ds)
leaf_test_ds = leaf_val_test_ds.take(leaf_val_batches // 2)
leaf_val_ds = leaf_val_test_ds.skip(leaf_val_batches // 2)

print("Training images:", int(leaf_train_ds.cardinality()))
print("Validating images:", int(leaf_val_ds.cardinality()))
print("Testing images:", int(leaf_test_ds.cardinality()))

print("\nFLOWER")
flower_train_ds, flower_val_test_ds = tf.keras.utils.image_dataset_from_directory(flower_directory, labels = "inferred", label_mode = "int", validation_split = 0.3,
                                                                                          subset = "both", color_mode = "rgb", shuffle = True, seed = 1, batch_size = 1)
flower_val_batches = tf.data.experimental.cardinality(flower_val_test_ds)
flower_test_ds = flower_val_test_ds.take(flower_val_batches // 2)
flower_val_ds = flower_val_test_ds.skip(flower_val_batches // 2)

print("Training images:", int(flower_train_ds.cardinality()))
print("Validating images:", int(flower_val_ds.cardinality()))
print("Testing images:", int(flower_test_ds.cardinality()))

dataset_names = [leaf_train_ds, leaf_val_ds, leaf_test_ds, flower_train_ds, flower_val_ds, flower_test_ds]

for name in dataset_names:
    labels_list = []
    for images, labels in name:
        labels_list.append(labels.numpy())
    all_labels = np.concatenate(labels_list)
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"\n{name} class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} images")