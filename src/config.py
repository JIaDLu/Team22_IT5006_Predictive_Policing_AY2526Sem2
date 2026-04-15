"""
Project configuration: paths, feature definitions, model hyperparameters.
"""
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ============================================================
# Data paths
# ============================================================
RAW_DATA_PATH = "data/exp_data/Chicago_Crimes_2015_2025.csv"
SCALER_PATH = "data/exp_data/scaler.pkl"
ADJACENCY_PATH = "data/exp_data/adjacency.npy"

# ============================================================
# Output paths
# ============================================================
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"

# ============================================================
# Temporal split boundaries
# ============================================================
TRAIN_START = "2015-01-01"
TRAIN_END = "2024-12-31"
VAL_START = "2025-01-01"
VAL_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2025-12-31"

# ============================================================
# Spatial constants
# ============================================================
NUM_REGIONS = 77  # Chicago Community Areas: 1-77
REGION_IDS = list(range(1, NUM_REGIONS + 1))

# ============================================================
# Feature engineering
# ============================================================
CRIME_TYPE_MAP = {
    "THEFT": "theft_count",
    "BATTERY": "battery_count",
}
LOCATION_TYPE_MAP = {
    "RESIDENCE": "residence_count",
    "STREET": "street_count",
    "APARTMENT": "apartment_count",
}

COUNT_FEATURES = [
    "crime_count",
    "theft_count",
    "battery_count",
    "residence_count",
    "street_count",
    "apartment_count",
]
TIME_FEATURES = ["day_of_week", "is_weekend", "month"]
FEATURE_COLS = COUNT_FEATURES + TIME_FEATURES
NUM_FEATURES = len(FEATURE_COLS)  # 9

TARGET_COL = "crime_count"
TARGET_IDX = 0

# ============================================================
# Sliding window
# ============================================================
WINDOW_SIZE = 7

# ============================================================
# Adjacency matrix
# ============================================================
KNN_K = 5

# ============================================================
# Model hyperparameters
# ============================================================
GNN_HIDDEN_DIM = 64
TEMPORAL_HIDDEN_DIM = 64
NUM_GNN_LAYERS = 2
DROPOUT = 0.1
MHA_NUM_HEADS = 4

# ============================================================
# Training
# ============================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
PATIENCE = 10
GRAD_CLIP_MAX_NORM = 5.0
HUBER_DELTA = 1.0

# ============================================================
# W&B
# ============================================================
WANDB_PROJECT = "lujiadong-nus/it5006"