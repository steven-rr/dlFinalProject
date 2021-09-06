import torch

# Setting this should help control memory pressure.
TENSOR_FLOAT_TYPE = torch.float32

# Root directory for plots
IMAGE_ROOT = './img'

DATA_DIR = './data'
CHARTS_DIR = './data/Charts'
CHARTS_CLOSE_DIR = './data/compressedChartsClose/CS7643_CHARTS'
CHARTS_EST_DIR = './data/compressedChartsEst/CS7643_CHARTS'
CHARTS_VOL_DIR = './data/compressedChartsVol/CS7643_CHARTS'
CHARTS_GRU_DIR = './data/compressedChartsGRU/CS7643_CHARTS'
CHART_FILE_EXT = '.png'
CHART_FILE_EXT_LEN = len(CHART_FILE_EXT)

# For labels, {0, 1, 2, 3, 4}
LABEL_BIN_THRESHOLDS = [-1.5, -0.5, 0, 0.5, 1.5]

# Do an 80/20 split
TRAIN_DATA_RATIO = 0.8

# Determines the modulo for logging epochs to the console (everything is logged to file).
EPOCH_MODULO = 1
# Determines the module for iteration index logged to the console while training an epoch (everything is logged to file).
ITERATION_MODULO = 10

# Define a root log directory
LOGS_DIR = './logs'

LOSS_DIR = './loss'

# Image Scale Factor
IMAGE_SCALE_FACTOR = 4

# Num Worker Threads for Data Load
DATA_LOADER_WORKERS = 4
