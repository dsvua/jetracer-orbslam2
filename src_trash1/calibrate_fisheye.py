import cv2
import numpy as np
import os
import glob

image_height = 720
image_width = 1280
# filename = "/home/serhiy/Downloads/images_temp/bin/good/10_output032.bin"
# left_image_np = np.fromfile(filename, dtype='uint8', count = image_width * image_height, offset = 0)
# right_image_np = np.fromfile(filename, dtype='uint8', count = image_width * image_height, offset = image_width * image_height)

# print(left_image_np.shape)
# print(image_width * image_height)

# left_image_cv = cv2.resize(left_image_np,(image_width,image_height))

# cv2.imshow('imageL',left_image_np.reshape(image_height, image_width))
# cv2.imshow('imageR',right_image_np.reshape(image_height, image_width))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
right_imgpoints = [] # 2d points in image plane.
left_imgpoints = [] # 2d points in image plane.
images = glob.glob('/home/serhiy/Downloads/images_temp/bin/good/*.bin')

for fname in images:
    left_image_np = np.fromfile(fname, dtype='uint8', count = image_width * image_height, offset = 0)
    right_image_np = np.fromfile(fname, dtype='uint8', count = image_width * image_height, offset = image_width * image_height)
    left_image_np = left_image_np.reshape(image_height, image_width)
    right_image_np = right_image_np.reshape(image_height, image_width)
    # Find the chess board corners
    left_ret, left_corners = cv2.findChessboardCorners(left_image_np, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    right_ret, right_corners = cv2.findChessboardCorners(right_image_np, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    if _img_shape == None:
        _img_shape = left_image_np.shape[:2]    # If found, add object points, image points (after refining them)
    if left_ret == True and right_ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(left_image_np,left_corners,(3,3),(-1,-1),subpix_criteria)
        left_imgpoints.append(left_corners)
        cv2.cornerSubPix(right_image_np,right_corners,(3,3),(-1,-1),subpix_criteria)
        right_imgpoints.append(right_corners)
N_OK = len(objpoints)
left_K = np.zeros((3, 3))
left_D = np.zeros((4, 1))
left_rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
left_tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
right_K = np.zeros((3, 3))
right_D = np.zeros((4, 1))
right_rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
right_tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
print("Image size ", right_image_np.shape[::-1])
print("calibrate right")
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        right_imgpoints,
        right_image_np.shape[::-1],
        right_K,
        right_D,
        right_rvecs,
        right_tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("calibrate left")
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        left_imgpoints,
        left_image_np.shape[::-1],
        left_K,
        left_D,
        left_rvecs,
        left_tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

print("Stereo calibrate")
retval, left_K, left_D, right_K, right_D, R, T = cv2.fisheye.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, left_K, left_D, right_K, right_D, right_image_np.shape[::-1])


print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(left_K.tolist()) + ")")
print("D=np.array(" + str(left_D.tolist()) + ")")

DIM = _img_shape[::-1]
print("initUndistortRectifyMap")
left_map1, left_map2 = cv2.fisheye.initUndistortRectifyMap(left_K, left_D, np.eye(3), left_K, DIM, cv2.CV_16SC2)
right_map1, right_map2 = cv2.fisheye.initUndistortRectifyMap(right_K, right_D, np.eye(3), right_K, DIM, cv2.CV_16SC2)

images = glob.glob('/home/serhiy/Downloads/images_temp/bin/good/*.bin')
for fname in images:
    left_image_np = np.fromfile(fname, dtype='uint8', count = image_width * image_height, offset = 0)
    right_image_np = np.fromfile(fname, dtype='uint8', count = image_width * image_height, offset = image_width * image_height)
    left_image_np = left_image_np.reshape(image_height, image_width)
    right_image_np = right_image_np.reshape(image_height, image_width)

    left_undistorted_img = cv2.remap(left_img, left_map1, left_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right_undistorted_img = cv2.remap(right_img, right_map1, right_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("left_undistorted " + fname, left_undistorted_img)
    cv2.imshow("right_undistorted " + fname, right_undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
