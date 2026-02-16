import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import keras
import os
from PIL import Image
import keras_tuner as kt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

print("SYSTEM/PACKAGE INFORMATION:")
print("TensorFlow version: ", tf.__version__)
print("keras version", keras.__version__)
print("GPU built with TensorFlow: ", tf.test.is_built_with_cuda())
print("Can access GPU: ", tf.config.experimental.list_physical_devices('GPU'))
print("TROUBLESHOOT DONE")

'''TRAIN-VALIDATE-TEST SPLIT'''
plant_categories_F = ["AAU", "ACO", "AMA", "BPU", "CFI", "CJA", "DRE", "IBI", "LLE", "LPU", "MDI",
"MPU", "MQU", "PDU", "PIN", "PSA", "PVU", "SAL", "SOB", "SOC", "SSA", "TIN"]
flower_directory = '/home/r2z103/res_dataset/Flower' # all png and resized to 224x224

print("DATASETS:")
print("FLOWER")
flower_train_ds, flower_val_test_ds = tf.keras.utils.image_dataset_from_directory(flower_directory, labels = "inferred", label_mode = "int", class_names = plant_categories_F,
                                                                                  validation_split = 0.3, subset = "both", image_size = (224, 224), color_mode = "rgb", shuffle = True, seed = 1, batch_size = None)
flower_val_batches = tf.data.experimental.cardinality(flower_val_test_ds)
flower_test_ds = flower_val_test_ds.take(flower_val_batches // 2)
flower_val_ds = flower_val_test_ds.skip(flower_val_batches // 2)

# Image Augmentation (for necessary classes)
augmentation_pipeline = [
    tf.keras.layers.RandomRotation([-0.20, 0.20]),
    tf.keras.layers.RandomFlip(), # horizontal_and_vertical by default
    tf.keras.layers.RandomBrightness((0.8, 1.2))
]

def conditional_augmentation(image, label, target):

    def augment():
        augmented = image
        for layer in augmentation_pipeline:
            augmented = layer(augmented, training=True)
        return augmented, label

    def no_augment():
        return image, label

    return tf.cond(
        tf.equal(label, target),
        augment,
        no_augment
    )

def count_dataset(ds):
    return sum(1 for _ in ds)

flower_class2_ds = flower_train_ds.filter(lambda img, lbl: tf.equal(lbl, 2))
flower_class14_ds = flower_train_ds.filter(lambda img, lbl: tf.equal(lbl, 14))

flower_class2_aug_ds = flower_class2_ds.map(
    lambda img, lbl: conditional_augmentation(img, lbl, 2),
    num_parallel_calls=tf.data.AUTOTUNE
)
flower_train_ds = flower_train_ds.concatenate(flower_class2_aug_ds)
flower_train_ds = flower_train_ds.shuffle(100)

flower_class14_aug_ds = flower_class14_ds.map(
    lambda img, lbl: conditional_augmentation(img, lbl, 14),
    num_parallel_calls=tf.data.AUTOTUNE
)
flower_train_ds = flower_train_ds.concatenate(flower_class14_aug_ds)
flower_train_ds = flower_train_ds.shuffle(100)

dataset_names = [flower_train_ds, flower_val_ds, flower_test_ds]

for name in dataset_names:
    labels_list = []
    for images, labels in name:
        labels_list.append(labels.numpy())
    all_labels = np.array(labels_list)
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"\n{name} class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} images")

flower_train_ds = flower_train_ds.batch(32).prefetch(buffer_size = tf.data.AUTOTUNE)
flower_val_ds = flower_val_ds.batch(32).prefetch(buffer_size = tf.data.AUTOTUNE)
flower_test_ds = flower_test_ds.batch(32).prefetch(buffer_size = tf.data.AUTOTUNE)

for images, labels in flower_train_ds.take(1):
    print("Images:", images.shape)
    print("Labels:", labels.shape)

'''MODEL PROPER'''
IMG_SIZE = (224, 224, 3) # Set image size

# Load base model
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SIZE, include_top=False, weights='imagenet')
# Freeze the base ResNet-50
base_model.trainable = False

# Preprocessing and main model
model = tf.keras.Sequential([
    base_model, # include_top may be False in the base_model if the immediately preceeding layer before it differs in input_shape. keep true if they are the same
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dense(22) # output layer; modified for 22 layers
    ])

# Compile model
for layer in model.layers[1:]: # skip base resnet50 model
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.summary()

# TRAIN-VAL
number_epochs = 30
train_val_history = model.fit(flower_train_ds,
                              epochs = number_epochs,
                              validation_data = (flower_val_ds)
                              )

model.save("flower_CNN-1.keras")
tf.keras.utils.plot_model(model, to_file = "flower_1.png", show_shapes = True, show_dtype = False, show_layer_names = False, show_layer_activations = False, show_trainable = False)
tf.keras.utils.plot_model(model, to_file = "flower_1_simple.png", show_shapes = True, show_dtype = True, show_layer_names = True, show_layer_activations = True, show_trainable = True)

# Evaluate train-val results
acc = train_val_history.history['accuracy']
val_acc = train_val_history.history['val_accuracy']
loss = train_val_history.history['loss']
val_loss = train_val_history.history['val_loss']

# Visualize train-val results
epochs_range = range(number_epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# BAYESIAN OPTIMIZATION

# Load saved CNN
tuned_model = tf.keras.models.load_model("flower_CNN-1.keras")
tuned_model.summary()

class FlowerHyperModel(kt.HyperModel):

    def build(self, hp):

        base_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = False

        fine_tune_at = hp.Int( # number of fine tuned layers
            'fine_tune_layers',
            min_value=20,
            max_value=len(base_model.layers),
            step=20
        )

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=True)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
        x = tf.keras.layers.Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
        x = tf.keras.layers.Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
        x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
        x = tf.keras.layers.Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(x)

        outputs = tf.keras.layers.Dense(25)(x)

        model = tf.keras.Model(inputs, outputs)

        learning_rate = hp.Choice('learning_rate', [1e-5, 1e-4, 1e-3]) # learning rate

        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = True

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        return model

    def fit(self, hp, model, *args, **kwargs): # batch size
        return model.fit(*args, batch_size=hp.Choice('batch_size', [16, 32, 64]), **kwargs)

tuner = kt.BayesianOptimization(FlowerHyperModel(), objective='val_accuracy', max_trials = 10, project_name='LeafCNN-1-finetune')
tuner.search(flower_train_ds, epochs=30, validation_data=flower_val_ds)

# Get and print optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best fine-tuned layers:", best_hps.get('fine_tune_layers'))
print("Best learning rate:", best_hps.get('learning_rate'))
print("Best dropout rate:", best_hps.get('dropout_rate'))
print("Best batch size:", best_hps.get('batch_size'))