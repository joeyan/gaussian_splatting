# (TODO move these to an options struct for reproducability)

# INITIALIZATION OPTIONS
INITIAL_OPACITY = 0.2  # initial opacity value (sigmoid activation)
INITIAL_SCALE_NUM_NEIGHBORS = 3  # number of neighbors used to compute initial scale
INITIAL_SCALE_FACTOR = (
    0.8  # scaling factor for initial scale:  log(mean_neighbor_dist * factor = initial_scale)
)
MAX_INITIAL_SCALE = 0.1  # maximum initial scale value

# SPLAT OPTIONS
NEAR_THRESH = 0.3
MH_DIST = 3.0
CULL_MASK_PADDING = 100
SATURATED_PIXEL_VALUE = 255.0

# ADAPTIVE CONTROL OPTIONS
DELETE_OPACITY_THRESHOLD = 0.1
CLONE_SCALE_THRESHOLD = 0.01
MAX_SCALE_NORM = 0.5

UV_GRAD_TRHRESHOLD = 0.0002  # paper = 0.0002

USE_FRACTIONAL_DENSIFICATION = True  # False to match paper
UV_GRAD_PERCENTILE = 0.96
SCALE_NORM_PERCENTILE = 0.99  # 1.0 to match paper

USE_SPLIT = True
USE_CLONE = True
USE_DELETE = True
ADAPTIVE_CONTROL_START = 1500
ADAPTIVE_CONTROL_END = 27500
ADAPTIVE_CONTROL_INVERVAL = 300

# keeps the reset opacity after the adaptive control
# works for reasonable number of iterations (<100k)
RESET_OPACITY_INTERVAL = 3001
RESET_OPACITY_VALUE = 0.15
RESET_OPACTIY_START = 1050
RESET_OPACITY_END = 27500

USE_SH_COEFF = True
MAX_SH_BAND = 3  # Can use bands 0, 1, 2, 3
ADD_SH_BAND_INTERVAL = 1000

SPLIT_SCALE_FACTOR = 1.6
NUM_SPLIT_SAMPLES = 2

# TRAIN OPTIONS
BASE_LR = 0.002
XYZ_LR_MULTIPLIER = 0.1
QUAT_LR_MULTIPLIER = 2
SCALE_LR_MULTIPLIER = 5
OPACITY_LR_MULTIPLIER = 10
RGB_LR_MULTIPLIER = 2
SH_LR_MULTIPLIER = 0.1

NUM_ITERS = 30000
SSIM_RATIO = 0.2

TEST_SPLIT_RATIO = 8  # same as paper and Mip-Nerf 360
TEST_EVAL_INTERVAL = 500

# DEBUG OPTIONS
SAVE_DEBUG_IMAGE_INTERVAL = 200
PRINT_INTERVAL = 100
PRINT_DEBUG_TIMING = False
OUTPUT_DIR = "splat_output"

# DATASET
DATASET_PATH = "/home/joe/Downloads/garden"
DOWNSAMPLE_FACTOR = 4

# Load model from checkpoint
LOAD_MODEL = False
MODEL_PATH = "splat_output/gaussians.pt"
