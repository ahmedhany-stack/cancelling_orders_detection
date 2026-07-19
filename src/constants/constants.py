# ===========================
# Data Split
# ===========================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ===========================
# Dataset
# ===========================

TARGET_COLUMN = "Cancelled"

# ===========================
# Model Training
# ===========================

CV = 5
N_JOBS = -1
PRIMARY_METRIC = "f1"

# ===========================
# Missing Values
# ===========================

NUMERIC_IMPUTER = "median"
CATEGORICAL_IMPUTER = "most_frequent"