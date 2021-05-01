// Frame preprocessing
#define PYRAMID_LEVELS 1
#define PYRAMID_MIN_LEVEL 0
#define PYRAMID_MAX_LEVEL PYRAMID_LEVELS

// FAST detector parameters
#define FAST_EPSILON (13.0f)
#define FAST_MIN_ARC_LENGTH 12
#define FAST_SCORE SUM_OF_ABS_DIFF_ON_ARC

// Remark: the Rosten CPU version only works with
//         SUM_OF_ABS_DIFF_ON_ARC and MAX_THRESHOLD

// NMS parameters
#define HORIZONTAL_BORDER 0
#define VERTICAL_BORDER 0
#define CELL_SIZE_WIDTH 32
#define CELL_SIZE_HEIGHT 32

#define DETECTOR_BASE_NMS_SIZE 3
#define MINIMUM_BORDER 3
#define FEATURE_DETECTOR_HORIZONTAL_BORDER 8
#define FEATURE_DETECTOR_VERTICAL_BORDER 8
#define FEATURE_DETECTOR_CELL_SIZE_WIDTH 32
#define FEATURE_DETECTOR_CELL_SIZE_HEIGHT 32

#define FAST_GPU_USE_LOOKUP_TABLE 1
#define FAST_GPU_USE_LOOKUP_TABLE_BITBASED 1
