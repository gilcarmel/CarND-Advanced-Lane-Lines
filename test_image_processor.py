import argparse
import glob
import itertools

import cv2

import lane_finder as lf


def read_images(path):
    """
    Read sample images under test_images
    :return: list of (filename, image) pairs
    """
    imgs = []
    fnames = glob.glob("{}/*.jpg".format(path))
    for fname in fnames:
        # Convert to RGB (pipline handles RGB to streamline video processing)
        rgb_image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        imgs.append(rgb_image)
    print("Read {} images.".format(len(imgs)))
    return zip(fnames, imgs)


def request_reprocess():
    global intermediate_images
    intermediate_images = None


def update_param(param_key, trackbar_value):
    param_def = lf.param_defs[param_key]
    # Convert from trackbar value (min 0, integer step) to actual value
    lf.params[param_key] = trackbar_value * param_def.step + param_def.min_value
    request_reprocess()


def create_slider(param_key, window_name):
    param_def = lf.param_defs[param_key]
    param_value = lf.params[param_key]
    trackbar_max = actual_to_trackbar_value(param_def, param_def.max_value)
    cv2.createTrackbar(
        param_def.description,
        window_name,
        actual_to_trackbar_value(param_def, param_value),
        trackbar_max,
        lambda param, k=None, state=None: update_param(param_key, param))


def display_image(window_name, img, *param_keys):
    """
    Display an image in a window and create sliders for controlling the image
    :param window_name: window name
    :param img: image to display
    :param param_keys: parameter keys for sliders
    :return:
    """
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    img = lf.to_bgr_if_necessary(img)
    cv2.imshow(window_name, img)

    # Create a slider for each parameter
    for param_key in param_keys:
        create_slider(param_key, window_name)


def actual_to_trackbar_value(param_def, value):
    return int((value - param_def.min_value) / param_def.step)


def add_warp_src_indicators(img):
    # Add indicators showing the warping source coordinates
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    src = lf.get_perspective_src(img)
    img = cv2.line(img, src[0], src[1], (255,0,0), 1)
    img = cv2.line(img, src[0], src[2], (255,0,0), 1)
    img = cv2.line(img, src[2], src[3], (255,0,0), 1)
    img = cv2.line(img, src[1], src[3], (255,0,0), 1)
    return img


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some test jpgs in an interactive UI')
    parser.add_argument("path", help="path containing test image jpgs")
    args = parser.parse_args()

    # read images and prepare to cycle through them
    images = read_images(args.path)
    image_cycle = itertools.cycle(images)

    filename, image = next(image_cycle)
    intermediate_images = None

    # Main loop
    while True:
        if not intermediate_images:
            lane, intermediate_images = lf.process_image(image)
            lf.write_output(filename, image, intermediate_images)
            display_image("Original", image)
            # display_image("Undistorted", output[UNDISTORTED])
            # display_image("HLS", output[HLS])
            # display_image("S", output[HLS])
            # display_image(SOBEL_X, output[SOBEL_X], SOBEL_X_KERNEL_SIZE, SOBEL_X_MIN, SOBEL_X_MAX)
            # display_image(S_THRESH, output[S_THRESH], S_MIN, S_MAX)
            combined_with_warp_src = add_warp_src_indicators(intermediate_images[lf.COMBINED_BINARY])
            display_image(
                lf.COMBINED_BINARY,
                combined_with_warp_src,
                lf.FAR_LEFT,
                lf.FAR_RIGHT,
                lf.NEAR_LEFT,
                lf.NEAR_RIGHT)
            display_image(lf.TOP_DOWN, intermediate_images[lf.TOP_DOWN])
            # display_image(lf.BOTTOM_HALF_HIST, output[lf.BOTTOM_HALF_HIST])
            # display_image(lf.LANE_LINE_POINTS, output[lf.LANE_LINE_POINTS])
            display_image(lf.LANE_LINE_POLYS, intermediate_images[lf.LANE_LINE_POLYS])
            display_image(lf.LANE_FILL, intermediate_images[lf.LANE_FILL])
            display_image(lf.FRONT_CAM_WITH_LANE_FILL, intermediate_images[lf.FRONT_CAM_WITH_LANE_FILL])

        key = cv2.waitKey(33)
        if key == ord('q'):
            # quit
            break
        elif key == ord('n'):
            # next image
            filename, image = next(image_cycle)
            intermediate_images = None
