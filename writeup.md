## Advanced Lane Finding Project 
---

The project is mainly composed of 2 files, the Jupyter notebook “Project 2 – Advanced Lane Finding” and the second file is the Lane Finder. 

The Lane Finder is composed of two objects. First, the line object which defines the attributes of each line that will define the lane. Second, the LaneFinder Object which consists of all the attributes, functions and the application pipeline grouping all those functions together.

The Jupyter Notebook is for initializing the folders, the image function, the Video function and the main.  The image function is used for testing the application pipeline on test images files. The video function is testing the pipeline but on the videos in the Input_Videos folder.

Jumping to the pipeline where all the actions happen, it is mainly composed of seven steps as shown in detail below:

Step 1:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

You can find the functions in this step under Section 1 in LaneFinder.py. it is composed of 3 functions, one function for Calibrating the camera and 2 getters functions for the chessboard and the Calibration output.

The camera calibration is using the CV2 library camera calibration.  The calibration is done on the list of chessboard files in the camera_cal folder first.

Step 2:
* Apply a distortion correction to raw images.

You can find the function in this step under Section 2 in LaneFinder.py. It is only composed of one function for the camera distortion correctness.


Step 3:
* Use color transforms, gradients, etc., to create a thresholded binary image.

You can find the functions in this step under Section 3 in LaneFinder.py. it is composed of 5 functions, 4 for calculating different thresholds: 
- Absolutes of the sobels
- Magnitude of the sobels together
- Direction of the line using the inverse tan
- HSL S channel threshold

The last function which is “apply_threshold” which is combining the thresholds together.


More details on the Sobels can be found in the following link: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html 
More details about the direction of the line using the inverse tan can be found in the following link: https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan.html 
More details on the HSL color scale can be found in the following link: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html

Step 4:
* Apply a perspective transform to rectify binary image ("birds-eye view").

You can find the function in this step under Section 4 in LaneFinder.py. It is only composed of one function for the warp_prespective using python library cv2. This function is helping to have a bird eye view of the road ahead of the car. 

More information about cv2.warpPresective can be found in the following link: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html

Step 5:
* Detect lane pixels and fit to find the lane boundary.

You can find the function in this step under Section 5 in LaneFinder.py. This part is composed of 5 functions, they are implemented to define a specific region of interest as per the output of the last step. Then we can define the line, or the windows that fit the lane lines. 


Step 6:
* Determine the curvature of the lane and vehicle position with respect to center.


You can find the function in this step under Section 6 in LaneFinder.py. 
To define the curvatures of the lane lines, there are 2 functions used one will show the details of the curvature as per the middle of the screen.  Those details will be added in the next step and will be written in the output images or videos.


Step 7:
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

You can find the function in this step under Section 7 in LaneFinder.py
This part is for adding the text to the frame and doing another warpd of the image to get it back to its original.

[//]: # (Image References)

[image1]: ./Pictures/undistort_output.png "Undistorted"
[image7]: ./Pictures/chessboard.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image8]: ./Pictures/thresholded_output.png “Thesholded output”
[image3]: ./Pictures/binary_combo_example.jpg "Binary Example"
[image4]: ./Pictures/ waped_image.jpg "Warp Example"
[image5]: ./Pictures/color_fit_lines.jpg "Fit Visual"
[image6]: ./Pictures/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is can be found in calibrate_camera in the LaneFinder.py. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt][./Pictures/image7]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is an example of undistorted image. Distortion correction is implemented in the distortion_correction in LaneFinder.py.

![alt][image1]


#### 2. Thresholds

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 146 through 202 in `LaneFinder.py`).  Here's an example of my output for this step.  


![alt][image8]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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

![alt][image4]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

