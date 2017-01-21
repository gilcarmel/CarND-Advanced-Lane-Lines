
# maximum allowed difference between second degree coefficients for polynomials to be considered parallel
POLY_THRESH_2 = 0.0005
# maximum allowed difference between first degree coefficients for polynomials to be considered parallel
POLY_THRESH_1 = 0.02
# lines with a curve radius greater than this are considered straight
CURVE_RADIUS_STRAIGHT_THRESH = 1000
# lines that have a curve need to have a (min radius / max radius) ratio above this
# to be considered "same curvature"
SAME_CURVATURE_RATIO_THRESH = 0.5


class Lane:
    """
    Data about a detected lane (left line, right line, confidence, etc)
    """

    def __init__(self, left_line, right_line, is_full_search):

        self.left_line = left_line
        self.right_line = right_line
        self.curve_ratio = -1
        self.curvature_radius = -1
        self.confident = False
        self.is_full_search = is_full_search

        if left_line.is_detected and right_line.is_detected:
            self.lane_width = self.right_line.x - self.left_line.x
            self.confident = self.is_confident()
            self.width = right_line.x - left_line.x

    def is_roughly_parallel(self):
        """
        Return True if lane lines are roughly parallel
        :return:
        """
        coeff_2_diff = self.left_line.polynomial_fit_m[0] - self.right_line.polynomial_fit_m[0]
        if abs(coeff_2_diff) > POLY_THRESH_2:
            return False
        coeff_1_diff = self.left_line.polynomial_fit_m[1] - self.right_line.polynomial_fit_m[1]
        if abs(coeff_1_diff) > POLY_THRESH_1:
            return False
        return True

    def similar_curve_radius(self):
        """
        Check with lane lines have a similar curve radisu
        :return: True if similar, False otherwise
        """
        l_radius = self.left_line.radius_of_curvature
        r_radius = self.right_line.radius_of_curvature
        # If the lines are basically straight, consider them parallel
        if self.is_straight():
            # If basically straight, make it official
            self.curvature_radius = CURVE_RADIUS_STRAIGHT_THRESH
            self.curve_ratio = 1
            return True

        self.curvature_radius = (l_radius + r_radius) * 0.5
        self.curve_ratio = min(l_radius, r_radius) / max(l_radius,r_radius)
        return self.curve_ratio > SAME_CURVATURE_RATIO_THRESH

    def is_confident(self):
        """
        Return true if we are confident in this prediction
        :return: True if confident, False otherwise
        """
        if not self.left_line.is_detected or not self.right_line.is_detected:
            return False

        # Don't short-circuit on false, because these methods calculate
        # various interesting metrics
        result = True

        # Reject anything that is way off standard lane width
        if not self.is_lane_width_reasonable():
            result = False

        # Reject lane lines that are not close to parallel
        if not self.is_roughly_parallel():
            result = False

        # Reject lane lines that don't have a similar curvature radius at
        # the bottom of the image
        if not self.similar_curve_radius():
            result = False

        return result

    def is_lane_width_reasonable(self):
        return 3.2 < self.lane_width < 4.2

    def is_straight(self):
        l_radius = self.left_line.radius_of_curvature
        r_radius = self.right_line.radius_of_curvature
        return l_radius > CURVE_RADIUS_STRAIGHT_THRESH \
            and r_radius > CURVE_RADIUS_STRAIGHT_THRESH

