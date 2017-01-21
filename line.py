import numpy as np


class Line:
    """
    Store the characteristics for a detected lane line in an image
    """
    def __init__(self, x_meters_per_pixel, y_meters_per_pixel, bottom_y):
        # meters per pixel (for calculate curvature radius from image)
        self.x_meters_per_pixel = x_meters_per_pixel
        self.y_meters_per_pixel = y_meters_per_pixel
        # bottom of image in pixels, representing car's current location
        self.bottom_y = bottom_y
        self.bottom_y_meters = self.bottom_y * self.y_meters_per_pixel
        # x value of the line bottom in meters
        self.x = None
        # x value of the line bottom in pixels
        self.x_pixels = None
        # detected points forming the line in pixels
        self.lane_points = None
        # minimum (aka top) y value for detected lane line
        self.min_y = None
        # polynomial coefficient fitting lane_points
        self.polynomial_fit = None
        # detected points forming the line in meters
        self.lane_points_m = None
        # polynomial coefficient fitting lane_points_m
        self.polynomial_fit_m = None
        # radius of curvature of the line in meters
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # set to true when lane_points are set.
        self.is_detected = False

    def set_lane_points(self, lane_points):
        """
        Sets the detected points on the lane line and calculates derived values,
        such as polygon fit and curvature, based on that.
        :param lane_points: tupes of points in pixel units
        :return:
        """
        self.lane_points = lane_points
        # If points were found, calculate derived values
        if len(self.lane_points) > 0:
            self.min_y = np.min(self.lane_points[:, 1])
            self.polynomial_fit = fit_poly(self.lane_points)
            self.lane_points_m = scale_points(self.lane_points, self.x_meters_per_pixel, self.y_meters_per_pixel)
            self.polynomial_fit_m = fit_poly(self.lane_points_m)
            self.x = np.polyval(self.polynomial_fit_m, self.bottom_y_meters)
            self.x_pixels = np.polyval(self.polynomial_fit, self.bottom_y)
            self.calculate_curve_radius()
            self.is_detected = True

    def calculate_curve_radius(self):
        """
        Calculate the curvature radius in meters
        :return:
        """
        polynomial = self.polynomial_fit_m
        self.radius_of_curvature = \
            ((1 + (2 * polynomial[0] * self.bottom_y_meters +
                   polynomial[1]) ** 2) ** 1.5) \
                / np.absolute(2 * polynomial[0])

        # If the lane is curving left, negate the radius
        if polynomial[0] < 0:
            self.radius_of_curvature *= -1


def fit_poly(points):
    """
    Returns a second degree polynomial fitting the points such that f(y) = x
    :param points: points
    :return: polynomial coefficients
    """
    if len(points) == 0:
        return None
    yvals = points[:, 1]
    xvals = points[:, 0]
    # Fit a second order polynomial to each fake lane line
    return np.polyfit(yvals, xvals, 2)


def scale_points(points, x_scale, y_scale):
    """
    Return a scaled copy of the points passed in
    :param points: points in pixels
    :param x_scale: meters per pixel in x direction
    :param y_scale: meters per pixel in y direction
    :return: scaled points array
    """
    scaled_points = np.copy(points)
    scaled_points[:, 0] *= x_scale
    scaled_points[:, 1] *= y_scale
    return scaled_points
