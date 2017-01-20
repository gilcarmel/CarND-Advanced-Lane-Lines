import os

import cv2
import imageio

imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
import lane_finder
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def generate_output_frame(img):
    global frame_number
    global basename
    global lanes
    lane, image_dict = lane_finder.process_image(img)
    lanes.append(lane)
    final_image = image_dict[lane_finder.FRONT_CAM_WITH_LANE_FILL]
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
        (0, 5000))


def plot_confident():
    y = [l.confident for l in lanes]
    b = [True for _ in lanes]
    plot_if_true(y, b, (-0.1, 2), "Confident?", "is_confident")


def plot_confident_width():
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
    y = [l.curvature_radius for l in lanes]
    confident = [l.confident for l in lanes]
    plot_if_true(y, confident, (0, 1000), "Confident curve radius", "confident radius value")


def plot_curvature_ratio():
    """
    Plot the percent difference between the left and right curve radius
    :return:
    """
    values = [l.curve_ratio for l in lanes]
    plot_values(values, (0, 1), "Curvature ratio", "curvature_ratio")


def plot_if_true(values, booleans, y_range, label, filename_suffix):
    y = [v for v,b in zip(values,booleans) if b]
    x = [index for index, b in enumerate(booleans) if b]
    plot_values(y, y_range, label, filename_suffix, x)

def plot_values(values, y_range, label, filename_suffix, x=None):
    fig = Figure()
    plot = fig.add_subplot(111)
    if x is None:
        plot.plot(values)
    else:
        plot.plot(x, values, ',')
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
    :return:
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


def plot_left_right_values(left_values, right_values, left_label, right_label, filename_suffix, y_range):
    fig = Figure()
    plot = fig.add_subplot(111)
    plot.plot(left_values)
    plot.plot(right_values)
    plot.set_ylim(y_range[0], y_range[1])
    plot.legend([left_label, right_label], loc='upper left')
    # Convert the plot to an image (i.e. numpy array)
    img = lane_finder.write_figure_to_array(fig)
    filename = os.path.join('intermediate/', basename, basename + '_' + filename_suffix + '.jpg')
    cv2.imwrite(filename, img)


if __name__ == "__main__":
    basename = "project_video"
    clip = VideoFileClip(basename + ".mp4")
    output_name = basename + "_out.mp4"
    frame_number = 0
    lanes = []
    # output_clip = clip.subclip(0,0.5).fl_image(generate_output_frame)
    output_clip = clip.fl_image(generate_output_frame)
    output_clip.write_videofile(output_name, audio=False)

    plot_curvature_radius()
    plot_curvature_ratio()
    plot_positions()
    plot_coefficients()
    plot_confident()
    plot_confident_width()
    plot_confident_parallel()
    plot_confident_curve_radius()
    plot_confident_radius_values()
