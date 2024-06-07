import os


# Result dataset directory
SPLIT_INPUT_PATH = "./dataset/split"
# Output directory
OUTPUT_PATH = "./output"

# Training testing, validation
TRAIN_PATH = os.path.sep.join([SPLIT_INPUT_PATH, "training"])
VAL_PATH = os.path.sep.join([SPLIT_INPUT_PATH, "validation"])
TEST_PATH = os.path.sep.join([SPLIT_INPUT_PATH, "testing"])

# Data splitting
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# Parameters
CLASSES = ["benign","malignant"]
BATCH_SIZE = 24
INIT_LR = 1e-3
EPOCHS = 20


