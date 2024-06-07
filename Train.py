import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from global_avg_pooling_CNN_Resnet import BC_Model

# Define paths to training, validation, and testing directories
train_dir = config.TRAIN_PATH
val_dir = config.VAL_PATH
test_dir = config.TEST_PATH

# Count the number of .png files in a directory and its subdirectories
total_train_images = len(glob.glob(os.path.join(train_dir, '**', '*.png'), recursive=True))
total_val_images = len(glob.glob(os.path.join(val_dir, '**', '*.png'), recursive=True))
total_test_images = len(glob.glob(os.path.join(test_dir, '**', '*.png'), recursive=True))

# Retrieve all training files
all_train_files = glob.glob(os.path.join(train_dir, '**', '*.png'), recursive=True)

# Calculate class weights based on the number of images in each class
train_labels = [int(os.path.basename(os.path.dirname(p))) for p in all_train_files]
train_labels = to_categorical(train_labels)
class_totals = train_labels.sum(axis=0)
class_weights = {i: class_totals.max() / class_totals[i] for i in range(len(class_totals))}

def plot_training_history(history, epochs, plot_path):
    """
    Plot training and validation accuracy and loss.
    
    Args:
        history (History): History object from model fitting.
        epochs (int): Number of epochs.
        plot_path (str): Path to save the plot.
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(plot_path)

# Data augmentation for training
train_aug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

# Initialize training data generator
train_generator = train_aug.flow_from_directory(
    train_dir,
    class_mode="categorical",
    target_size=(50, 50),
    color_mode="rgb",
    shuffle=True,
    batch_size=config.BATCH_SIZE
)

# Data augmentation for validation and testing
val_aug = ImageDataGenerator(rescale=1 / 255.0)

# Initialize validation data generator
val_generator = val_aug.flow_from_directory(
    val_dir,
    class_mode="categorical",
    target_size=(50, 50),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BATCH_SIZE
)

# Initialize testing data generator
test_generator = val_aug.flow_from_directory(
    test_dir,
    class_mode="categorical",
    target_size=(50, 50),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BATCH_SIZE
)

# Build the model
print("Building the model...")
model = BC_Model.build(width=50, height=50, depth=3, classes=2)

# Hyperparameters
epochs = 20
initial_learning_rate = 0.0001

# Compile the model
print("Compiling the model...")
optimizer = Adam(learning_rate=initial_learning_rate, decay=initial_learning_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Model checkpoint callback
checkpoint_path = os.path.join(config.OUTPUT_PATH, "custom_weights.hdf5")
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint_callback]

# Train the model
print("Training the model...")
history = model.fit(
    x=train_generator,
    steps_per_epoch=total_train_images // config.BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=total_val_images // config.BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    epochs=epochs
)

# Load the best model
best_model_path = os.path.join(config.OUTPUT_PATH, 'custom_weights.hdf5')
best_model = load_model(best_model_path)

# Predict on the test data
print("Predicting on the test data...")
predictions = best_model.predict(x=test_generator, steps=(total_test_images // config.BATCH_SIZE) + 1)
predicted_classes = np.argmax(predictions, axis=1)

# Plot training history
plot_training_history(history, epochs, config.PLOT_PATH_CM)
