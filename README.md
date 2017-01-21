
[//]: # (Image References)

[chessboard]
[chessboard
[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

**Advanced Lane Finding Project**

This project uses computer vision techniques to detect lane information from a car's front-facing camera. From an input video clip, it produces an output video marked and annotated with lane info. I completed this project as part of Udacity's Self-Driving Car nanodegree program.

There are two distinct layers of processing: a lane-finding pipeline for single images, and a video layer that leverages temporal coherence (i.e. detections from previous frames) to smooth the output and optimize the performance of the single image layer. 

# Single Image Processing

The single image processing pipeline can be summarized as follows:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Undistort raw image.
* Detect lane separator "candidate pixels" using gradients and color transforms on the source image.
* Apply a perspective transform to generate a bird's-eye view of the candidate pixels.
* Find the left and right lane separator lines in the bird's-eye view and fit a polynomial to each
* Perform a confidence check on the detection
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The source code can be found in lane_finder.py TODO
Here is the top-level function that drives the pipeline. TODO

I was inspired by TODO to create an interactive UI for tweaking the various parameters contained in this pipeline. Here is a short video demonstrating its use. TODO

---
Each step is described in detail below.

###Camera Calibration

Before processing an image, we need to account for lens distortion (i.e. fisheye effect from the camera). OpenCV includes utilities for determining a camera's calibration parameters, which quantify how the camera distorts images. Since all the images are taken with the same camera, we only need to do this once.

cv2.calibrateCamera() calculates calibration parameters given a set of 3D points in world space and their corresponding 2D locations in the image. We use a chessboard pattern photographed from several angles to generate input as follows:
* 2D image points are detected by cv2.findChessboardCorners().
* 3D points are generated using a regular grid on the (x,y), with z=0.

[chessboard] Distorted image with corners detected
[undistorted_chessboard] Undistorted image

The code can be found [here TODO](.)

### Undistort raw image
We use cv2.undistort() to undistort the input image using the calibration parameters we calculated.

Here's the original image and the undistorted image. TODO

Code: TODO

### Detect lane separator "candidate pixels"
We generate a binary image with pixels that are good bets to be part of the lane separator painted white: TODO image

This image is obtained by ORing together three separate images - 
A thresholded Sobel filter in the X direction on the grayscale image
A thresholded Sobel filter in the X direction on the S channel of the HLS color space
A threshold on the S channel of the HLS color space

Each of these identified some pixels that other components missed, so combining them together yeilded a better detection of candidate pixels. 

Code is here TODO

This is one of the weaker parts of my pipeline - it does the job on the project video but does not perform well on the challenge videos.

### Generate a bird's-eye view

Now we perform a perspective warp on the image from the previous step, to bring it into a bird's eye view that can be used to find the lane lines. OpenCV's cv2.getPerspectiveTransform() can generate such a transformation given a quadrilateral on the source image and its desired location on the destination image:

Source image with 4 points
Destination image with 4 points

The source points are hard coded into the pipeline based on a measurement of one "vanilla" frame of the video. Unfortunately I found that the locations of the source source quad is extremely sensitive to small changes in the car's attitute (i.e. small bounces pointing the camera up or down). An incorrect source quad will cause the lane lines to appear skewed in the top-down view, which reduces our confidence in the detection: TODO

Code is here TODO

### Find the left and right lane lines

Now we search for the left and right lane lines. The algorithm is as follows:
* For each lane line (left and right):
  * Divide the image into ten horizontal bands. From the bottom, for each band:
    * Create search window centered horizontally above the the previous band's point
    * Calculate the center of mass of white pixels in the search window
    * Add the center of mass to the list of points
  TODO: image
  * Fit a second degree polynomial to all the detected points
  TODO: image

### Determine the curvature and vehicle position
  
### Perform a confidence check on the detection
To determine whether we are confident in the prediction, we detected lane lines must be:
* Roughly 3.7 meters apart (per US standards)
* Roughly parallel (2nd and 1st order polynomial coefficients are within a threshold)
* Roughly the same curvature radius 
Sample images TODO
Code is here TODO
### Warp back onto the original image.
We filled the region between the left and right lane lines, and warp it back as if it is seen from the front-facing camera (again using cv2.getPerspectiveTransform() but this time with reversed source and destination quadrilaterals)
### Annotate image with lane curvature and vehicle position.


####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

