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
    global left_lines
    global right_lines
    left_line, right_line, image_dict, confident = lane_finder.process_image(img)
    left_lines.append(left_line)
    right_lines.append(right_line)
    final_image = image_dict[lane_finder.FRONT_CAM_WITH_LANE_FILL]
    if frame_number % 10 == 0:
        lane_finder.write_output("{0}/frame_{1:0>4}".format(basename, frame_number), img, image_dict)
        bgr_final_image = lane_finder.to_bgr_if_necessary(final_image)
        cv2.imwrite('intermediate/{0}/{1:0>4}.jpg'.format(basename, frame_number), bgr_final_image)
    frame_number += 1
    return final_image


def plot_curvature_radius():
    """
    Plot the curvature radius of the left and right lines over the entire clip
    :return:
    """
    left_line_values = [l.radius_of_curvature for l in left_lines]
    right_line_values = [l.radius_of_curvature for l in right_lines]
    plot_left_right_values(
        left_line_values,
        right_line_values,
        'left line curvature (m)',
        'right line curvature (m)',
        'curvatures',
        (0,5000))


def plot_curvature_radius_diff_pct():
    """
    Plot the percent difference between the left and right curve radius
    :return:
    """
    left_line_values = [l.radius_of_curvature for l in left_lines]
    right_line_values = [l.radius_of_curvature for l in right_lines]
    max_min_radius = [(max(r[0], r[1]), min(r[0],r[1])) for r in zip(left_line_values, right_line_values)]
    pct_diff = [(r[0] - r[1]) / (r[1]) * 100 for r in max_min_radius]
    fig = Figure()
    plot = fig.add_subplot(111)
    plot.plot(pct_diff)
    plot.set_ylim(0, 200)
    plot.legend(["Curvature difference (%)"], loc='upper left')
    filename_suffix = "curvature_diff"
    # Convert the plot to an image (i.e. numpy array)
    img = lane_finder.write_figure_to_array(fig)
    filename = os.path.join('intermediate/', basename, basename + '_' + filename_suffix + '.jpg')
    cv2.imwrite(filename, img)


def plot_positions():
    """
    Plot the curvature radius of the left and right lines over the entire clip
    :return:
    """
    left_line_values = [l.x for l in left_lines]
    right_line_values = [l.x for l in right_lines]
    plot_left_right_values(
        left_line_values,
        right_line_values,
        'left line position (m)',
        'right line position (m)',
        'positions',
        (-2,8))


def plot_coefficients():
    """
    Plot the polynomial coefficients of the left and right lines over the entire clip
    :return:
    """
    left_line_values = [l.polynomial_fit_m[0] for l in left_lines]
    right_line_values = [l.polynomial_fit_m[0] for l in right_lines]
    plot_left_right_values(
        left_line_values,
        right_line_values,
        'left line 2nd degree coeff ',
        'right line 2nd degree coeff ',
        'second_deg',
        (-0.002, 0.002))

    left_line_values = [l.polynomial_fit_m[1] for l in left_lines]
    right_line_values = [l.polynomial_fit_m[1] for l in right_lines]
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
    left_lines = []
    right_lines = []
    output_clip = clip.fl_image(generate_output_frame)
    output_clip.write_videofile(output_name, audio=False)

    plot_curvature_radius()
    plot_curvature_radius_diff_pct()
    plot_positions()
    plot_coefficients()
