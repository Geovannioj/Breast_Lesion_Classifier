import os

ORIG_INPUT_DATASET = "datasets/origin"
BASE_PATH = "datasets/output"

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.5