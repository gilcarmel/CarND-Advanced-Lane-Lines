"""Advanced lane finding project. """

import glob
import itertools
import os.path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Keys into intermidiate image dictionary
UNDISTORTED = '00_undistorted'
HLS = '01_hls'
S = '02_s'
S_THRESH = '03_s_thresh'  # Thresholded by S channel
GRAY = '04_grey'
SOBEL_X = '05_sobel_x'  # Thresholded by sobel in X direction
COMBINED_BINARY = '06_combined_binary'  # Combined thresholded images
TOP_DOWN = '07_top_down'  # top down view
BOTTOM_HALF_HIST = '08_bottom_half_hist'  # histogram of bottom half of the image
LANE_LINE_POINTS = '09_lane_line_points'
LANE_LINE_POLYS = '10_lane_line_polynomials'

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
    SOBEL_X_KERNEL_SIZE: 5,
    SOBEL_X_MIN: 30,
    SOBEL_X_MAX: 100,
    S_MIN: 170,
    S_MAX: 255,
    FAR_LEFT: 0.359,
    FAR_RIGHT: 0.655,
    NEAR_LEFT: 0.137,
    NEAR_RIGHT: 0.876,
}


def read_images():
    """
    Read sample images under test_images
    :return: list of (filename, image) pairs
    """
    imgs = []
    fnames = glob.glob("test_images/*.jpg")
    for fname in fnames:
        imgs.append(cv2.imread(fname))
    print("Read {} images.".format(len(imgs)))
    return zip(fnames, imgs)


def undistort_image(img):
    """
    Undistort an image using the global calibration parameters
    :param img: input image
    :return: undistorted image
    """
    global camera_matrix, distortion_coeffs
    return cv2.undistort(img, camera_matrix, distortion_coeffs, None, camera_matrix)


def convert_to_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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
    s_thresh = imgs_dict[S_THRESH]
    combined = np.zeros_like(sobel_x, dtype=np.uint8)
    combined[(sobel_x > 0) | (s_thresh > 0)] = 255
    return combined


def perspective_projection(img):
    w = img.shape[1]
    h = img.shape[0]
    src = get_perspective_src(img)

    # Define 4 corners for top-down view
    top_down_left = w * 0.25
    top_down_right = w * 0.75
    dst = np.float32([[top_down_left, h * 0.75],
                      [top_down_right, h * 0.75],
                      [top_down_left, h],
                      [top_down_right, h]])
    # get the transform matrix
    matrix = cv2.getPerspectiveTransform(np.float32(src), dst)
    # Warp to a top-down view
    return cv2.warpPerspective(img, matrix, img.shape[::-1])


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


def find_lane_lines_in_bands(img, histogram):
    """
    Find points on the left and right lane lines by searching
    stacked vertical bands.

    :param img: thresholded, top-down view
    :param histogram: histogram of full image along x-axis
    :return: (lefts, rights). lefts and rights are np.arrays of
      points on each lane line.
    """
    half_width = int(img.shape[1] / 2)
    left_peak = np.argmax(histogram[:half_width])
    right_peak = half_width + np.argmax(histogram[half_width:])
    num_bands = 10
    search_window_half_width_ratio = 0.05
    lefts = search_climbing_bands(img, left_peak, num_bands, search_window_half_width_ratio)
    rights = search_climbing_bands(img, right_peak, num_bands, search_window_half_width_ratio)
    return np.array(lefts), np.array(rights)


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
    Divide the image into vertical bands and predict the lane position in each band
    :param img: input image (thresholded top-down view of lane lines)
    :param start_x: x coordinate to start search in lowest band
    :param num_bands: number of bands
    :param search_window_half_width_ratio: width of the search area
      (as a ratio of img width)
    :return:  list of (x,y) tuples representing the lane-line prediction for
      each band
    """
    height = int(img.shape[0])
    width = int(img.shape[1])
    band_height = int(float(height) / num_bands)
    cur_bottom = height - 1
    peaks = []
    search_window_half_width = int(search_window_half_width_ratio * width)
    for _ in np.arange(0, num_bands):
        # Generate a histogram for a small window centered
        # on the lane detected for the previous band
        search_left_start = max(0, start_x - search_window_half_width)
        search_left_end = min(width - 1, start_x + search_window_half_width)
        search_window = img[
                        cur_bottom - band_height: cur_bottom,
                        search_left_start:search_left_end
                        ]
        histogram = np.sum(search_window, axis=0)
        max_index = np.argmax(histogram)
        # Only update the detection if the maximum is a non-zero value
        if histogram[max_index] > 0:
            start_x = max_index + search_left_start
            peaks.append((start_x, cur_bottom - int(band_height / 2)))
        cur_bottom = max(0, cur_bottom - band_height)
    return peaks


def find_histogram(img):
    """
    Calculate the histogram of pixel values along th x-axis
    (i.e. the sum of pixel values for each column of the image)
    :param img: input image
    :return: histogram
    """
    half_height = int(img.shape[0] / 2)
    return np.sum(img[half_height:, :], axis=0)


def draw_lane_line_points(img, lefts, rights):
    """
    Return an image with circles representing points in
    the detected left/right lane lines
    :param img: input image
    :param lefts: left lane line point tuples
    :param rights: right lane line point tuples
    :return: img with circles super-imposed
    """
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for left in lefts:
        cv2.circle(img, (left[0], left[1]), 5, (255, 0, 0), -1)
    for right in rights:
        cv2.circle(img, (right[0], right[1]), 5, (0, 0, 255), -1)
    return img


def fit_polynomials(lefts, rights):
    """
    Returns a second-order polynomial fitting a points on the left
      and right lane lines
    :param lefts: points on the left lane line
    :param rights: points on the right lane line
    :return: (left_poly, right_poly) - second order polynomial coefficients
    """
    return fit_poly(lefts), fit_poly(rights)


def fit_poly(points):
    """
    Returns a second degree polynomial fitting the points such that f(y) = x
    :param points: points
    :return: polynomial coefficients
    """
    yvals = points[:, 1]
    xvals = points[:, 0]
    # Fit a second order polynomial to each fake lane line
    left_fit = np.polyfit(yvals, xvals, 2)
    return left_fit


def draw_lane_line_polynomials(img, left_poly, right_poly):
    """
    Draw polynomials on an image
    :param img: image
    :param left_poly: left lane line polynomial
    :param right_poly: right lane line polynomial
    :return: new image with polynomials drawn on it
    """
    img = np.copy(img)
    draw_polyline(img, left_poly)
    draw_polyline(img, right_poly)
    return img


def draw_polyline(img, poly):
    """
    Draw a polynomial on the image passed in
    :param img: image
    :param poly: polynomial coefficients
    :return: None
    """
    yvals = np.linspace(0, img.shape[0] - 1, num=100)
    xvals = poly[0] * yvals ** 2 + poly[1] * yvals + poly[2]
    points = np.int32([xvals, yvals]).T
    cv2.polylines(img, [points], False, (0, 255, 0), 3)


def process_image(original):
    """
    Perform lane finding on an image
    :param original: input image
    :return: Dictionary containing output image and all intermediate images
    """
    result = {UNDISTORTED: undistort_image(original)}
    result[HLS] = convert_to_hls(result[UNDISTORTED])
    result[S] = result[HLS][:, :, 2]
    result[GRAY] = convert_to_gray(result[UNDISTORTED])
    result[SOBEL_X] = thresholded_sobel_x(result[GRAY])
    result[S_THRESH] = threshold_image(result[S], params[S_MIN], params[S_MAX])
    result[COMBINED_BINARY] = combined_binary(result)
    result[TOP_DOWN] = perspective_projection(result[COMBINED_BINARY])
    histogram = find_histogram(result[TOP_DOWN])
    result[BOTTOM_HALF_HIST] = plot_histogram_to_array(histogram)
    lefts, rights = find_lane_lines_in_bands(result[TOP_DOWN], histogram)
    result[LANE_LINE_POINTS] = draw_lane_line_points(result[TOP_DOWN], lefts, rights)
    l_poly, r_poly = fit_polynomials(lefts,rights)
    result[LANE_LINE_POLYS] = draw_lane_line_polynomials(
        result[LANE_LINE_POINTS],
        l_poly,
        r_poly)
    return result


def calibrate_camera():
    """
    Calculate camera calibration parameters using the images in camera_cal/
    :return: (camera_matrix, distortion_coefficients)
    """

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    gray = None

    for fname in glob.glob('camera_cal/*.jpg'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the corners in every image.
        # The grid is cropped in some images, so try several
        # combinations of row/column counts
        # TODO: figure out why calibration4.jpg always fails
        for num_rows, num_columns in [(6, 9), (5, 9), (6, 8), (7, 6)]:
            ret, corners, objp = find_corners(gray, num_columns, num_rows)
            if ret:
                obj_points.append(objp)
                img_points.append(corners)
                break

    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
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


def display_image(window_name, img, *param_keys):
    """
    Display an image in a window and create sliders for controlling the image
    :param window_name: window name
    :param img: image to display
    :param param_keys: parameter keys for sliders
    :return:
    """
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, img)

    # Create a slider for each parameter
    for param_key in param_keys:
        create_slider(param_key, window_name)


def request_reprocess():
    global output
    output = None


def update_param(param_key, trackbar_value):
    param_def = param_defs[param_key]
    # Convert from trackbar value (min 0, integer step) to actual value
    params[param_key] = trackbar_value * param_def.step + param_def.min_value
    request_reprocess()


def create_slider(param_key, window_name):
    param_def = param_defs[param_key]
    param_value = params[param_key]
    trackbar_max = actual_to_trackbar_value(param_def, param_def.max_value)
    cv2.createTrackbar(
        param_def.description,
        window_name,
        actual_to_trackbar_value(param_def, param_value),
        trackbar_max,
        lambda param, k=None, state=None: update_param(param_key, param))


def actual_to_trackbar_value(param_def, value):
    return int((value - param_def.min_value) / param_def.step)


def add_warp_src_indicators(img):
    # Add indicators showing the warping source coordinates
    src = get_perspective_src(img)
    img = cv2.line(img, src[0], src[1], 255, 4)
    img = cv2.line(img, src[0], src[2], 255, 4)
    img = cv2.line(img, src[2], src[3], 255, 4)
    img = cv2.line(img, src[1], src[3], 255, 4)
    return img


def write_output(orig_filename, orig, output_images):
    dirname, basename = os.path.split(orig_filename)
    out_path = os.path.join('intermediate', dirname, basename.replace(".jpg", ""))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    cv2.imwrite('{}/00_orig.jpg'.format(out_path), orig)
    for output_key in output_images:
        cv2.imwrite('{}/{}.jpg'.format(out_path, output_key), output_images[output_key])


if __name__ == "__main__":
    # determine camera calibration parameters
    camera_matrix, distortion_coeffs = calibrate_camera()

    # read images and prepare to cycle through them
    images = read_images()
    image_cycle = itertools.cycle(images)

    filename, image = next(image_cycle)
    output = None

    # Main loop
    while True:
        if not output:
            output = process_image(image)
            write_output(filename, image, output)
            display_image("Original", image)
            # display_image("Undistorted", output[UNDISTORTED])
            # display_image("HLS", output[HLS])
            # display_image("S", output[HLS])
            # display_image(SOBEL_X, output[SOBEL_X], SOBEL_X_KERNEL_SIZE, SOBEL_X_MIN, SOBEL_X_MAX)
            # display_image(S_THRESH, output[S_THRESH], S_MIN, S_MAX)
            combined_with_warp_src = add_warp_src_indicators(output[COMBINED_BINARY])
            display_image(COMBINED_BINARY, output[COMBINED_BINARY], FAR_LEFT, FAR_RIGHT, NEAR_LEFT, NEAR_RIGHT)
            display_image(TOP_DOWN, output[TOP_DOWN])
            display_image(BOTTOM_HALF_HIST, output[BOTTOM_HALF_HIST])
            display_image(LANE_LINE_POINTS, output[LANE_LINE_POINTS])
            display_image(LANE_LINE_POLYS, output[LANE_LINE_POLYS])

        key = cv2.waitKey(33)
        if key == ord('q'):
            # quit
            break
        elif key == ord('n'):
            # next image
            filename, image = next(image_cycle)
            output = None