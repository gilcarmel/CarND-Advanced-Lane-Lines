""" This is the main image processing code for finding lanes.
 process_image() is the main entry point - it executes a CV pipeline to find the left and right lane
 lines on an image.
 NOTE: upon being imported this module determines the camera calibration based on the calibration images.
"""

import glob
import os.path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import ndimage

# Keys into intermidiate image dictionary
from lane import Lane
from line import Line

UNDISTORTED = '00_undistorted'
HLS = '01_hls'
S = '02_s'
H = '02_h'
L = '02_l'
S_THRESH = '03_s_thresh'  # Thresholded by S channel
GRAY = '04_grey'
SOBEL_X = '05_sobel_x'  # Thresholded by sobel in X direction
SOBEL_X_S = '05_sobel_x_s'  # S channel thresholded by sobel in X direction
COMBINED_BINARY = '06_combined_binary'  # Combined thresholded images
TOP_DOWN = '07_top_down'  # top down view
BOTTOM_HALF_HIST = '08_bottom_half_hist'  # histogram of bottom half of the image
LANE_LINE_POINTS = '09_lane_line_points'
LANE_LINE_POLYS = '10_lane_line_polynomials'
LANE_FILL = '11_lane_fill_region'
# undistorted front camera with lane-fill overlay
FRONT_CAM_WITH_LANE_FILL = '12_front_cam_with_lane_fill'
ANNOTATED_IMAGE = '12_annotated'

# KEYS into paramter dictionaries
SOBEL_X_KERNEL_SIZE = 'SOBEL_X_KERNEL_SIZE'
SOBEL_X_MIN = 'SOBEL_X_MIN'
SOBEL_X_MAX = 'SOBEL_X_MAX'
S_MIN = 'S_MIN'
S_MAX = 'S_MAX'
FAR_LEFT = 'FAR_LEFT'
FAR_RIGHT = 'FAR_RIGHT'
NEAR_LEFT = 'NEAR_LEFT'
NEAR_RIGHT = 'NEAR_RIGHT'


# Definition of a pipeline parameter
class ParamDef(object):
    def __init__(self, min_value, max_value, step, description):
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.description = description
        pass


# Definition of all parameters in our pipeline
param_defs = {
    SOBEL_X_KERNEL_SIZE: ParamDef(3, 15, 2, "Sobel X kernel size"),
    SOBEL_X_MIN: ParamDef(0, 255, 1, "Sobel min threshold"),
    SOBEL_X_MAX: ParamDef(0, 255, 1, "Sobel max threshold"),
    S_MIN: ParamDef(0, 255, 1, "S min threshold"),
    S_MAX: ParamDef(0, 255, 1, "S max threshold"),
    # Source box for warping
    FAR_LEFT: ParamDef(0, 1, 0.001, "Far left %"),
    FAR_RIGHT: ParamDef(0, 1, 0.001, "Far right %"),
    NEAR_LEFT: ParamDef(0, 1, 0.001, "Near left %"),
    NEAR_RIGHT: ParamDef(0, 1, 0.001, "Near right %"),
}

# Parameters to use for various steps of the pipeline
params = {
    SOBEL_X_KERNEL_SIZE: 7,
    SOBEL_X_MIN: 22,
    SOBEL_X_MAX: 100,
    S_MIN: 139,
    S_MAX: 255,
    FAR_LEFT: 0.374,
    FAR_RIGHT: 0.647,
    NEAR_LEFT: 0.165,
    NEAR_RIGHT: 0.879,
}


def undistort_image(img, cam_matrix, distortion_coefficients):
    """
    Undistort an image using the global calibration parameters
    :param img: input image
    :param cam_matrix: camera projection matrix
    :param distortion_coefficients: distortion coefficients
    :return: undistorted image
    """
    return cv2.undistort(img, cam_matrix, distortion_coefficients, None, cam_matrix)


def convert_to_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def thresholded_sobel_x(gray):
    """
    Returns a thresholded image based on the sobel derivative in the X direction
    :param gray: input grayscale image
    :return: thresholded image
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=params[SOBEL_X_KERNEL_SIZE])
    abssx = np.absolute(sobelx)  # Absolute to accept light->dark and dark->light
    # Normalize and convert to 8 bit
    scale_factor = np.max(abssx) / 255
    scaled_sobel = (abssx / scale_factor).astype(np.uint8)
    # Threshold
    thresh_min = params[SOBEL_X_MIN]
    thresh_max = params[SOBEL_X_MAX]
    return threshold_image(scaled_sobel, thresh_min, thresh_max)


def threshold_image(img, thresh_min, thresh_max):
    """
    Returns a thresholded version of a one-channel image
    :param img: source image
    :param thresh_min: minimum value
    :param thresh_max: maximum value
    :return: thresholded binary image
    """
    thresholded_img = np.zeros_like(img, dtype=np.uint8)
    thresholded_img[(img >= thresh_min) & (img <= thresh_max)] = 255
    return thresholded_img


def combined_binary(imgs_dict):
    """
    Create a combined binary image based on previous intermediate images
    :param imgs_dict: dictionary of intermediate images
    :return: combined binary image
    """
    sobel_x = imgs_dict[SOBEL_X]
    sobel_x_s = imgs_dict[SOBEL_X_S]
    s_thresh = imgs_dict[S_THRESH]
    combined = np.zeros_like(sobel_x, dtype=np.uint8)
    combined[(sobel_x > 0) | (sobel_x_s > 0) | (s_thresh > 0)] = 255
    return combined


def perspective_projection(img, to_top_down=True):
    top_down, front_facing = get_perspective_map_regions(img)
    # get the transform matrix
    if to_top_down:
        matrix = cv2.getPerspectiveTransform(front_facing, top_down)
    else:
        matrix = cv2.getPerspectiveTransform(top_down, front_facing)
    # Warp
    flipped_shape = img.shape[0:2][::-1]
    return cv2.warpPerspective(img, matrix, flipped_shape)


def get_perspective_map_regions(img):
    """
    Get the source and dest quadrilaterals defining a perspective projection
    :param img: image (used for its size)
    :return: quads on images from top-down cam, front-facing cam
    """
    w = img.shape[1]
    h = img.shape[0]
    front_facing_quad = np.float32(get_perspective_src(img))
    # Define 4 corners for top-down view
    top_down_left = w * 0.25
    top_down_right = w * 0.75
    top_down_quad = np.float32(
      [[top_down_left, h * 0.75],
       [top_down_right, h * 0.75],
       [top_down_left, h],
       [top_down_right, h]])
    return top_down_quad, front_facing_quad


def get_perspective_src(img):
    """
    Returns the source coordinates for a perspective warp
    :param img: image
    :return: source coordinates
    """
    # Return (bottom, far left, far right
    width = img.shape[1]
    height = img.shape[0]
    far_left = int(width * params[FAR_LEFT])
    far_right = int(width * params[FAR_RIGHT])
    top = int(height * 0.744)  # somewhat arbitrary top where other params were tuned
    near_left = int(width * params[NEAR_LEFT])
    near_right = int(width * params[NEAR_RIGHT])
    bottom = height - 1
    return [(far_left, top), (far_right, top), (near_left, bottom), (near_right, bottom)]


def find_lane_lines_in_bands(img, left_line, right_line, prev_left_x=None, prev_right_x=None):
    """
    Find points on the left and right lane lines by searching
    stacked vertical bands.

    :param img: thresholded, top-down view
    :param left_line: Upon return, left_line.lane_points will contain detected points
    :param right_line: Upon return, left_line.lane_points will contain detected points
    :param prev_left_x: previous x position of left lane line (None if no previous line detected)
    :param prev_right_x: previous x prosition of right lane line (None if no previous line detected)
    """
    num_bands = 10
    search_window_half_width_ratio = 0.05
    left_line.set_lane_points(*search_climbing_bands(img, prev_left_x, num_bands, search_window_half_width_ratio))
    right_line.set_lane_points(*search_climbing_bands(img, prev_right_x, num_bands, search_window_half_width_ratio))


def plot_histogram_to_array(histogram):
    """
    Plots a histogram to an in-memory image
    :param histogram: histogram to plot
    :return: numpy array representing an image
    """
    fig = Figure()
    plot = fig.add_subplot(111)
    plot.plot(histogram)
    # Convert the plot to an image (i.e. numpy array)
    return write_figure_to_array(fig)


def write_figure_to_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    data = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def search_climbing_bands(img, start_x, num_bands, search_window_half_width_ratio):
    """
    Divide the image into vertical bands and predict the lane line position in each band
    :param img: input image (thresholded top-down view of lane lines)
    :param start_x: x coordinate to start search in lowest band
    :param num_bands: number of bands
    :param search_window_half_width_ratio: width of the search area
      (as a ratio of img width)
    :return:  (detected lane-line points, search windows)
    """
    height = int(img.shape[0])
    width = int(img.shape[1])
    band_height = int(float(height) / num_bands)
    search_bottom = height - 1
    points = []  # Points detected
    search_window_half_width = int(search_window_half_width_ratio * width)
    search_windows = []
    for _ in np.arange(0, num_bands):
        # Create a search window above the previous band and centered horizontally
        # about the previous detection
        search_left = max(0, start_x - search_window_half_width)
        search_right = min(width - 1, start_x + search_window_half_width)
        search_top = max(0, search_bottom - band_height)
        search_window = img[
                        search_top: search_bottom,
                        int(search_left):int(search_right)
                        ]
        search_windows.append((search_left, search_right, search_top, search_bottom))
        # Find the center of mass within the search window and add it to the line
        center_of_mass_in_window = ndimage.measurements.center_of_mass(search_window)[::-1]
        # Empty iamges return nan for center of mass - ignore
        if not np.math.isnan(center_of_mass_in_window[0] + center_of_mass_in_window[1]):
            start_x = search_left + center_of_mass_in_window[0]
            center_of_mass_y = search_bottom - band_height + center_of_mass_in_window[1]
            points.append([start_x, center_of_mass_y])
        search_bottom = search_top
    return np.array(points), search_windows


def find_histogram(img):
    """
    Calculate the histogram of pixel values along th x-axis
    (i.e. the sum of pixel values for each column of the image)
    :param img: input image
    :return: histogram, left peak, right peak
    """
    half_height = int(img.shape[0] / 2)
    histogram = np.sum(img[half_height:, :], axis=0)
    half_width = int(img.shape[1] / 2)
    left_peak = np.argmax(histogram[:half_width])
    right_peak = half_width + np.argmax(histogram[half_width:])
    return histogram, left_peak, right_peak


def draw_lane_line_points(img, left_line, right_line):
    """
    Return an image with circles representing points in
    the detected left/right lane lines
    :param img: input image
    :param left_line: left lane line
    :param right_line: right lane line
    :return: img with circles super-imposed
    """
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for left in left_line.lane_points:
        cv2.circle(img, (int(left[0]), int(left[1])), 5, (255, 0, 0), -1)
    for search_window in np.int32(left_line.search_windows):
        cv2.rectangle(img, (search_window[0], search_window[2]), (search_window[1],search_window[3]), (200,65,139), 3)
    for right in right_line.lane_points:
        cv2.circle(img, (int(right[0]), int(right[1])), 5, (0, 0, 255), -1)
    for search_window in np.int32(right_line.search_windows):
        cv2.rectangle(img, (search_window[0], search_window[2]), (search_window[1],search_window[3]), (200,65,139), 3)
    return img


def draw_lane_line_polynomials(
    img,
    left_line,
    right_line
):
    """
    Draw polynomials on an image
    :param img: image
    :param left_line: Line object for left lane
    :param right_line: Line object for right lane
    :return: new image with polynomials drawn on it
    """
    img = np.copy(img)
    max_y = img.shape[0] - 1
    draw_polyline(img, left_line.polynomial_fit, left_line.min_y, max_y)
    draw_polyline(img, right_line.polynomial_fit, right_line.min_y, max_y)
    return img


def draw_polyline(img, poly, min_y, max_y):
    """
    Draw a polynomial on the image passed in
    :param img: image
    :param poly: polynomial coefficients
    :param min_y: y value for line start
    :param max_y: y value for line end
    :return: None
    """
    if poly is not None:
        points = get_poly_points(max_y, min_y, poly)
        cv2.polylines(img, [points], False, (0, 255, 0), 3)


def get_poly_points(max_y, min_y, poly):
    """
    Returns points along a polynomial
    (evaluated along 100 points from f(min_y) to f(max_y))
    :param max_y: maximum y to evaluate
    :param min_y: minimum y to evaluate
    :param poly: second order polynomial coefficients
    :return: points
    """
    yvals = np.linspace(min_y, max_y, num=100)
    xvals = np.polyval(poly, yvals)
    # noinspection PyUnresolvedReferences
    points = np.int32([xvals, yvals]).T
    return points


def draw_lane_fill_region(img, left_line, right_line, color=(100, 255, 0)):

    img = np.zeros_like(img)

    if left_line.polynomial_fit is None or right_line.polynomial_fit is None:
        return img

    max_y = img.shape[0] - 1
    left_points = get_poly_points(max_y, left_line.min_y, left_line.polynomial_fit)
    right_points = get_poly_points(max_y, right_line.min_y, right_line.polynomial_fit)[::-1]
    all_points = np.concatenate([left_points, right_points, left_points[:1]])
    cv2.fillPoly(img, [all_points], color)
    return img


def fill_lane_region(img, top_down_lane_fill):
    front_facing_lane_fill = perspective_projection(top_down_lane_fill, False)
    return cv2.addWeighted(img, 1., front_facing_lane_fill, 0.5, 0)


def process_image(original, previous_lane=None):
    """
    Perform lane finding on an image
    :param original: input image
    :param previous_lane: previous lane detection
    :return: (Lane, image dictionary for all pipeline steps)
    """
    left_line, right_line = create_lane_lines(original)

    image_dict = {UNDISTORTED: undistort_image(original, camera_matrix, distortion_coeffs)}
    image_dict[HLS] = convert_to_hls(image_dict[UNDISTORTED])
    image_dict[S] = image_dict[HLS][:, :, 2]
    image_dict[H] = image_dict[HLS][:, :, 0]
    image_dict[L] = image_dict[HLS][:, :, 1]
    image_dict[GRAY] = convert_to_gray(image_dict[UNDISTORTED])
    image_dict[SOBEL_X_S] = thresholded_sobel_x(image_dict[S])
    image_dict[SOBEL_X] = thresholded_sobel_x(image_dict[GRAY])
    image_dict[S_THRESH] = threshold_image(image_dict[S], params[S_MIN], params[S_MAX])
    image_dict[COMBINED_BINARY] = combined_binary(image_dict)
    image_dict[TOP_DOWN] = perspective_projection(image_dict[COMBINED_BINARY])

    # If we don't have a prior detetion, use histogram peaks to find
    # search starting points
    if previous_lane is None:
        histogram, left_search_x, right_search_x = find_histogram(image_dict[TOP_DOWN])
        image_dict[BOTTOM_HALF_HIST] = plot_histogram_to_array(histogram)
    # Otherwise start searching from the previous detection's x position
    else:
        left_search_x = previous_lane.left_line.x_pixels
        right_search_x = previous_lane.right_line.x_pixels
    find_lane_lines_in_bands(
        image_dict[TOP_DOWN],
        left_line, right_line,
        left_search_x,
        right_search_x)

    image_dict[LANE_LINE_POINTS] = draw_lane_line_points(image_dict[TOP_DOWN], left_line, right_line)
    image_dict[LANE_LINE_POLYS] = draw_lane_line_polynomials(
        image_dict[LANE_LINE_POINTS],
        left_line,
        right_line)
    image_dict[LANE_FILL] = draw_lane_fill_region(image_dict[LANE_LINE_POLYS], left_line, right_line)
    image_dict[FRONT_CAM_WITH_LANE_FILL] = fill_lane_region(
        image_dict[UNDISTORTED],
        image_dict[LANE_FILL])
    image_dict[ANNOTATED_IMAGE] = np.copy(image_dict[FRONT_CAM_WITH_LANE_FILL])

    lane = Lane(left_line, right_line, original.shape[1], previous_lane is None)
    annotate_image(image_dict[ANNOTATED_IMAGE], lane.curvature_radius, lane.center_offset)

    return lane, image_dict


def create_lane_lines(original):
    """
    Initializes two lane line structures
    :param original: original image
    :return: (left Line, right Line)
    """
    height = original.shape[0]
    # These are derived by dividing known distances by measured pixel distances in the top-down view
    ym_per_pix = 9.14 / 230  # meters per pixel in y dimension (9.14m is the distance between dashes on a lane line)
    xm_per_pix = 3.7 / 640  # meteres per pixel in x dimension (3.7m is the width of a lane)
    left_line = Line(xm_per_pix, ym_per_pix, height)
    right_line = Line(xm_per_pix, ym_per_pix, height)
    return left_line, right_line


def get_min_y_values(lefts, rights):
    """
    Get the minimum y values for the points identified. This
    corresponds to the highest point on the topdown image where
    we should extend the lane lines to.
    :param lefts: array of points on the left line
    :param rights: array of points on the right line
    :return: (min_left_y, min_right_y). Either can be None if points are empty
    """
    min_left_y = np.min(lefts[:, 1]) if len(lefts) > 0 else None
    min_right_y = np.min(rights[:, 1]) if len(rights) > 0 else None
    return min_left_y, min_right_y


def calibrate_camera():
    """
    Calculate camera calibration parameters using the images in camera_cal/
    :return: (camera_matrix, distortion_coefficients)
    """

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    gray = None

    output_path = 'camera_cal_output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for fname in glob.glob('camera_cal/*.jpg'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        basename = os.path.basename(fname)

        # Find the corners in every image.
        # The grid is cropped in some images, so try several
        # combinations of row/column counts until we get a successful detection
        for num_rows, num_columns in [(6, 9), (5, 9), (6, 8), (7, 6)]:
            ret, corners, objp = find_corners(gray, num_columns, num_rows)
            if ret:
                obj_points.append(objp)
                img_points.append(corners)
                cv2.drawChessboardCorners(img, (num_rows,num_columns), corners, ret)
                cv2.imwrite(os.path.join(output_path, basename), img)
                break
            else:
                cv2.drawChessboardCorners(img, (num_rows,num_columns), corners, ret)
                cv2.imwrite(os.path.join(output_path, basename+"_failed_{}_{}.jpg".format(num_rows,num_columns)), img)

    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    for fname in glob.glob('camera_cal/*.jpg'):
        basename = os.path.basename(fname)
        img = cv2.imread(fname)
        undistored = undistort_image(img, mtx, dist)
        cv2.imwrite(os.path.join(output_path, basename + "_undistorted.jpg"), undistored)

    return mtx, dist


def find_corners(gray, num_columns, num_rows):
    """
    Find corners in a grayscale images
    :param gray: image
    :param num_columns: number of grid columns
    :param num_rows:  number of grid rows
    :return: success, image corners, object corners
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((num_columns * num_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_rows, 0:num_columns].T.reshape(-1, 2)
    ret, corners = cv2.findChessboardCorners(gray, (num_rows, num_columns), None)

    # termination criteria for sub-pixel corner-finding
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners, objp


def write_output(orig_filename, orig, output_images):
    dirname, basename = os.path.split(orig_filename)
    out_path = os.path.join('intermediate', dirname, basename.replace(".jpg", ""))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    cv2.imwrite('{}/00_orig.jpg'.format(out_path), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
    for output_key in output_images:
        intermediate_image = output_images[output_key]
        intermediate_image = to_bgr_if_necessary(intermediate_image)
        cv2.imwrite('{}/{}.jpg'.format(out_path, output_key), intermediate_image)


def to_bgr_if_necessary(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def annotate_image(img, curvature_radius, center_lane_offset):
    radius = "{:.0f}".format(curvature_radius) if curvature_radius is not None else "?"
    cv2.putText(img, "radius = {} m".format(radius), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
    offset = "{:.2f}".format(center_lane_offset) if center_lane_offset is not None else "?"
    cv2.putText(img, "offset = {}m".format(offset), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

# determine camera calibration parameters
camera_matrix, distortion_coeffs = calibrate_camera()
