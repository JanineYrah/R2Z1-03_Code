import tensorflow as tf
from tensorflow import keras
import matplotlib as plt

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

directory = None #insert link to folder of images

train_val_dataset = tf.keras.utils.image_dataset_from_directory(directory, labels = 'inferred', label_mode = 'categorical',
                                                                image_size = (224, 224), shuffle = False, validation_split = 0.3, subset = 'both')
validate_ds, test_ds = tf.keras.utils.split_dataset(train_val_dataset, left_size=0.5, right_size=None, shuffle=False, seed=None)


'''MODEL PROPER'''
IMG_SIZE = (224, 224, 3) # Set image size

# Load base model
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SIZE,
include_top=False,
weights='imagenet')
# Freeze the base ResNet-50
base_model.trainable = False

# Preprocessing and main model
model = tf.keras.Sequential([
    tf.keras.Input(shape=IMG_SIZE), # still unsure on input_shape
	tf.keras.layers.Resizing(224, 224), # resizing layer
    
    '''Data augmentation layers could either  be pipeline or 
    individually. I'll do it individually here for now'''
    
    tf.keras.layers.RandomRotation(20),
    tf.keras.layers.RandomFlip(), # horizontal_and_vertical by default
    tf.keras.layers.RandomCrop(30, 30),
    tf.keraslayers.RandomBrightness((0.8, 1.2)),

	'''include_top may be False in the base_model if the immediately
    preceeding layer before it differs in input_shape. keep true if they are the same'''
    
    base_model,
    # Additional layers
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.Dense(25) # output layer; modified for 25 layers
    ])

'''# Extra: preprocessing pipeline, if we want to fuse all data augmentation into one na; Just replace the individual layers of model as this
preprocessing_pipeline = tf.keras.layers.Pipeline([
    tf.keras.layers.RandomRotation(20),
    tf.keras.layers.RandomFlip(), # horizontal_and_vertical by default
    tf.keras.layers.RandomCrop(30, 30),
    tf.keras.layers.RandomBrightness((0.8, 1.2))
])'''

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.summary()
model.save("leaf_CNN-1.keras")

# TRAIN-VAL
number_epochs = 30
train_val_history = model.fit(img_train, label_train,
                              epochs = number_epochs,
                              batch_size = 32,
                              validation_data = (x_val, y_val)
                              )

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
tuned_model = tf.keras.models.load_model("leaf_CNN-1.keras")

# Build hypermodel
def build_model(hp):
	# Define hyperparameter space
    model = tuned_model(
        batch_size=hp.Choice('batch_size', [16, 32, 64]),
        dropout_rate=hp.Choice('dropout_rate', min_value=0.1, max_value=0.5, step=0.1),
        # unsure for number of fine-tuned layers
    )

    hp_learning_rate=hp.Choice('learning_rate', [1e-5, 1e-4, 1e-3, 0.01, 0.1])

    # compile
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
    
    return model

tuner = keras.tuner.BayesianOptimization(build_model, objective='val_accuracy',
                     max_epochs=10)
tuner.search(img_train, label_train, epochs=30,
validation_split=0.2)

# Get and print optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print("Optimal batch size:", best_hps.get('batch_size'))
print("Optimal learning rate:", best_hps.get('learning_rate'))
print("Optimal dropout rate:", best_hps.get('dropout_rate'))