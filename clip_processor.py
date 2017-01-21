import os

import cv2
import imageio
import numpy

imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
import lane_finder
# noinspection PyUnresolvedReferences
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Start a full search after this many frames of not being confident
# about our lane detection
MAX_NOT_CONFIDENT = 10

# Use an average of all confident predictions from the smoothing window
# to generate a prediction for the current frame
SMOOTHING_WINDOW = 10


def make_final_image(image_dict):
    """
    Creates the final image by smoothing the confident predictions
    within the smoothing window and
    :param image_dict: intermediate images generated by the lane finder
    :return: final image
    """
    smoothed_prediction = get_smoothed_detection()
    if smoothed_prediction is None:
        return image_dict[lane_finder.UNDISTORTED]

    curvature_radius, center_lane_offset, lane_region = smoothed_prediction

    final_image = lane_finder.fill_lane_region(
        image_dict[lane_finder.UNDISTORTED],
        lane_region)

    lane_finder.annotate_image(final_image, curvature_radius, center_lane_offset)
    return final_image


def get_smoothed_detection():
    confident_lanes_in_smoothing_window = \
        [lane for frame, lane in confident_lanes[-SMOOTHING_WINDOW:] if frame_number - frame <= SMOOTHING_WINDOW]
    # No confident predictions in smoothing window
    if len(confident_lanes_in_smoothing_window) == 0:
        return None

    curvature_radius = numpy.mean([l.curvature_radius for l in confident_lanes_in_smoothing_window])
    center_lane_offset = numpy.mean([l.center_offset for l in confident_lanes_in_smoothing_window])
    return curvature_radius, center_lane_offset, last_confident_image_dict[lane_finder.LANE_FILL]


def previous_confident_lane():
    """
    Return the most recent confident lane prediction, if it hasn't expired
    :return:
    """

    if len(confident_lanes) == 0:
        # No previous confident lane detections
        return None

    last_confident_frame_number, last_confident_lane = confident_lanes[-1]
    # Most recent confident lane detection has not expired
    if frame_number - last_confident_frame_number < MAX_NOT_CONFIDENT:
        return last_confident_lane

    # Most recent confident lane detection has expired
    return None


def generate_output_frame(img):
    """
    Perform lane detection on one frame of input video
    :param img: input frame
    :return: output frame with lane overlay + info
    """
    global frame_number
    global basename
    global last_confident_image_dict

    lane, image_dict = lane_finder.process_image(img, previous_confident_lane())
    lanes.append(lane)
    if lane.confident:
        confident_lanes.append((frame_number, lane))
        last_confident_image_dict = image_dict

    final_image = make_final_image(image_dict)

    if frame_number % 10 == 0:
        lane_finder.write_output("{0}/frame_{1:0>4}".format(basename, frame_number), img, image_dict)
        bgr_final_image = lane_finder.to_bgr_if_necessary(final_image)
        cv2.imwrite('intermediate/{0}/{1:0>4}.jpg'.format(basename, frame_number), bgr_final_image)

    frame_number += 1
    return final_image


def left_lines():
    return [lane.left_line for lane in lanes]


def right_lines():
    return [lane.right_line for lane in lanes]


def plot_curvature_radius():
    """
    Plot the curvature radius of the left and right lines over the entire clip
    :return:
    """
    left_line_values = [l.radius_of_curvature for l in left_lines()]
    right_line_values = [l.radius_of_curvature for l in right_lines()]
    plot_left_right_values(
        left_line_values,
        right_line_values,
        'left line curvature (m)',
        'right line curvature (m)',
        'curvatures',
        (-5000, 5000),
        marker=',')


def plot_full_search():
    """
    Plot which frames required a full search for the line starting points
    due to not having a recent confident prediction
    :return: None
    """
    full_search = [l.is_full_search for l in lanes]
    plot_values(full_search, (-0.1, 1.3), "full_search", "full_search", marker=',')


def plot_confident():
    """
    Plot which lane detetions we are confident about
    :return: None
    """
    y = [l.confident for l in lanes]
    b = [True for _ in lanes]
    plot_if_true(y, b, (-0.1, 2), "Confident?", "is_confident")


def plot_confident_width():
    """
    Plot which lane detections have a width we are confident about
    :return:
    """
    y = [1 if l.is_lane_width_reasonable() else 0 for l in lanes]
    b = [True for _ in lanes]
    plot_if_true(y, b, (-0.1, 2), "Confident width?", "is_confident_width")


def plot_confident_parallel():
    values = [1 if l.is_roughly_parallel() else 0 for l in lanes]
    b = [True for _ in lanes]
    plot_if_true(values, b, (-0.1, 2), "Confident parallel?", "is_confident_parallel")


def plot_confident_curve_radius():
    values = [1 if l.similar_curve_radius() else 0 for l in lanes]
    b = [True for _ in lanes]
    plot_if_true(values, b, (-0.1, 2), "Confident radius?", "is_confident_radius")


def plot_confident_radius_values():
    y = [1.0 / l.curvature_radius for l in lanes]
    confident = [l.confident for l in lanes]
    plot_if_true(y, confident, (-0.01, 0.01), "inverse curve radius", "inv_curve_radius")


def plot_curvature_ratio():
    """
    Plot the percent difference between the left and right curve radius
    :return:
    """
    values = [l.curve_ratio for l in lanes]
    plot_values(values, (0, 1), "Curvature ratio", "curvature_ratio", marker=',')


def plot_if_true(values, booleans, y_range, label, filename_suffix):
    y = [v for v, b in zip(values, booleans) if b]
    x = [index for index, b in enumerate(booleans) if b]
    plot_values(y, y_range, label, filename_suffix, x, ',')


def plot_values(values, y_range, label, filename_suffix, x=None, marker=None):
    fig = Figure()
    plot = fig.add_subplot(111)
    if x is None:
        if marker is None:
            plot.plot(values)
        else:
            plot.plot(values, marker)
    else:
        plot.plot(x, values, marker)
    plot.set_ylim(y_range[0], y_range[1])
    plot.legend([label], loc='upper left')
    # Convert the plot to an image (i.e. numpy array)
    img = lane_finder.write_figure_to_array(fig)
    filename = os.path.join('intermediate/', basename, basename + '_' + filename_suffix + '.jpg')
    cv2.imwrite(filename, img)


def plot_positions():
    """
    Plot the curvature radius of the left and right lines over the entire clip
    :return:
    """
    left_line_values = [l.x for l in left_lines()]
    right_line_values = [l.x for l in right_lines()]
    plot_left_right_values(
        left_line_values,
        right_line_values,
        'left line position (m)',
        'right line position (m)',
        'positions',
        (-2, 8))


def plot_coefficients():
    """
    Plot the polynomial coefficients of the left and right lines over the entire clip
    :return: None
    """
    left_line_values = [l.polynomial_fit_m[0] for l in left_lines()]
    right_line_values = [l.polynomial_fit_m[0] for l in right_lines()]
    plot_left_right_values(
        left_line_values,
        right_line_values,
        'left line 2nd degree coeff ',
        'right line 2nd degree coeff ',
        'second_deg',
        (-0.002, 0.002))

    left_line_values = [l.polynomial_fit_m[1] for l in left_lines()]
    right_line_values = [l.polynomial_fit_m[1] for l in right_lines()]
    plot_left_right_values(
        left_line_values,
        right_line_values,
        'left line 1st degree coeff ',
        'right line 1st degree coeff ',
        'first_deg',
        (-0.1, 0.1))


def plot_left_right_values(left_values, right_values, left_label, right_label, filename_suffix, y_range, marker=None):
    """
    Plot values for the left and right lane line
    :param left_values: values for left lane line
    :param right_values: values for riht lane line
    :param left_label: label for left values
    :param right_label: label for right values
    :param filename_suffix: suffix of filename to save the image
    :param y_range: y range
    :param marker: marker shape code, e.g. '.' (plots a line if None)
    :return: None
    """
    fig = Figure()
    plot = fig.add_subplot(111)
    if marker is None:
        plot.plot(left_values)
        plot.plot(right_values)
    else:
        plot.plot(left_values, marker)
        plot.plot(right_values, marker)
    plot.set_ylim(y_range[0], y_range[1])
    plot.legend([left_label, right_label], loc='upper left')
    # Convert the plot to an image (i.e. numpy array)
    img = lane_finder.write_figure_to_array(fig)
    filename = os.path.join('intermediate/', basename, basename + '_' + filename_suffix + '.jpg')
    cv2.imwrite(filename, img)


def plot_metrics():
    """
    Plot charts that show how the predictions went over the entire clip
    :return: None
    """
    plot_curvature_radius()
    plot_curvature_ratio()
    plot_positions()
    plot_coefficients()
    plot_confident()
    plot_confident_width()
    plot_confident_parallel()
    plot_confident_curve_radius()
    plot_confident_radius_values()
    plot_full_search()


if __name__ == "__main__":
    basename = "project_video"
    clip = VideoFileClip(basename + ".mp4")
    output_name = basename + "_out.mp4"
    frame_number = 0
    last_confident_image_dict = None
    lanes = []
    confident_lanes = []
    # output_clip = clip.subclip(0,2.0).fl_image(generate_output_frame)
    output_clip = clip.fl_image(generate_output_frame)
    output_clip.write_videofile(output_name, audio=False)

    # Plot some metrics
    plot_metrics()
