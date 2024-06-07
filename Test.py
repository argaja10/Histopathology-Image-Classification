import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import config

# Paths to the training, validation, and testing directories
train_directory = config.TRAIN_PATH
validation_directory = config.VAL_PATH
test_directory = config.TEST_PATH

# Function to count the number of .png files in a directory
def count_png_files(directory_path):
    return len(glob.glob(os.path.join(directory_path, '**', '*.png'), recursive=True))

# Total number of images in each set
total_train_images = count_png_files(train_directory)
total_validation_images = count_png_files(validation_directory)
total_test_images = count_png_files(test_directory)

# Initialize the validation data augmentation object
validation_augmentor = ImageDataGenerator(rescale=1 / 255.0)

# Initialize the testing generator
test_generator = validation_augmentor.flow_from_directory(
    test_directory,
    class_mode="categorical",
    target_size=(50, 50),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BATCH_SIZE
)

# Load the best performing model
model_path = os.path.join(config.OUTPUT_PATH, 'custom _weights.hdf5')
loaded_model = load_model(model_path)

# Predict on the test data
print("Predicting on the test data...")
print(f"Total Train: {total_train_images}, Total Validation: {total_validation_images}, Total Test: {total_test_images}")
predictions = loaded_model.predict(x=test_generator, steps=(total_test_images // config.BATCH_SIZE) + 1)
predicted_classes = np.argmax(predictions, axis=1)

# Print the classification report
print(classification_report(test_generator.classes, predicted_classes, target_names=test_generator.class_indices.keys()))

# Compute confusion matrix and derive accuracy, sensitivity, and specificity
conf_matrix = confusion_matrix(test_generator.classes, predicted_classes)
total_samples = conf_matrix.sum()
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / total_samples
sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
specificity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

# Print confusion matrix, accuracy, sensitivity, and specificity
print(conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Function to display random images from the test set along with true and predicted classes
def display_random_test_images(generator, predicted_classes, num_images=100):
    class_labels = {v: k for k, v in generator.class_indices.items()}
    for _ in range(num_images):
        index = random.randint(0, len(generator.filenames) - 1)
        image_path = generator.filepaths[index]
        true_class = generator.classes[index]
        predicted_class = predicted_classes[index]

        # Load and display the image
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.title(f"True: {class_labels[true_class]}, Predicted: {class_labels[predicted_class]}")
        plt.axis('off')
        plt.show()

# Display random images from the test set
display_random_test_images(test_generator, predicted_classes, num_images=100)
