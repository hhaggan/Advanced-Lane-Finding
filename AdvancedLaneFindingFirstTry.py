#defining the list of libraries 
import os 
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import cv2
import logging

class AdvancedLaneFinderError(Exception):
    """Exception to be thrown in case of failure in AdvancedLaneFinding."""
    pass

class AdvanceLaneFinding:
    def __init__(self, image_size=(720, 1280), chessboard_image_dir='chessboard_images', absolute_sobel_x=(7, 15, 100), absolute_sobel_y=(7, 15, 100),
                 magnitude_sobel=(7, 30, 100),
                 direction_sobel=(31, 0.5, 1.0),
                 s_channel_thresh=(170, 255),
                 warp_perspective=(np.float32([(532, 496),
                                               (756, 496),
                                               (288, 664),
                                               (1016, 664)]),
                                   np.float32([(288, 366),
                                               (1016, 366),
                                               (288, 664),
                                               (1016, 664)])),
                 sliding_window_params=(8, 100, 50),
                 meters_per_pixel=(30/720, 3.7/700),
                 max_recent_xfitted=10,
                 lane_detection_failure_count_before_sliding_window=20,
                 region_of_interest_verts=np.array([[(0, 720),
                                                     (640, 405),
                                                     (640, 405),
                                                     (1280, 720)]],
                                                   dtype=np.int32),
                 ) -> None:
        if os.path.isdir(chessboard_image_dir):
            self._chessboard_image_dir = os.path.abspath(chessboard_image_dir)
            logging.info("Directory with calibration images: %s.", self._chessboard_image_dir)
        else:
            raise AdvancedLaneFinderError("%s directory does not exist." % chessboard_image_dir)

        # initialize list of calibration chessboard images
        self._chessboard_image_path_list = [os.path.join(self._chessboard_image_dir, fname)
                                            for fname in os.listdir(self._chessboard_image_dir)]
        if not self._chessboard_image_path_list:
            raise AdvancedLaneFinderError("No calibration images found in %s." % self._chessboard_image_dir)
        else:
            logging.info("There are %d calibration images.", len(self._chessboard_image_path_list))

        # set image size
        self._image_size = image_size
        if len(image_size) != 2:
            raise AdvancedLaneFinderError("Image size should be a pair of (height, width), but got %s." % image_size)
        logging.info("Advanced lane finder accepts images with resolution %s.", self._image_size)

        # image size, calibration matrix, distortion coefficients, rotation vectors, and translation vectors
        # will be initialized in self.calibrate_camera()
        self._calibration_matrix = None
        self._distortion_coefficients = None
        self._rotation_vectors = None
        self._translation_vectors = None
        self._calibrated = False
        logging.info("Camera calibration will happen only once, while the first image is processed.")

        # thresholds (absolute Sobel_x)
        if absolute_sobel_x[1] > absolute_sobel_x[2]:
            raise AdvancedLaneFinderError(
                "Thresholds for absolute Sobel_x operator are incorrect; minimum greater than maximum [ %s ]."
                % absolute_sobel_x)
        self._abs_sobel_x_kernel = absolute_sobel_x[0]
        self._abs_sobel_x_thresh_min = absolute_sobel_x[1]
        self._abs_sobel_x_thresh_max = absolute_sobel_x[2]

        # thresholds (absolute Sobel_y)
        if absolute_sobel_y[1] > absolute_sobel_y[2]:
            raise AdvancedLaneFinderError(
                "Thresholds for absolute Sobel_y operator are incorrect; minimum greater than maximum [ %s ]."
                % absolute_sobel_y)
        self._abs_sobel_y_kernel = absolute_sobel_y[0]
        self._abs_sobel_y_thresh_min = absolute_sobel_y[1]
        self._abs_sobel_y_thresh_max = absolute_sobel_y[2]

        # thresholds (magnitude Sobel)
        if magnitude_sobel[1] > magnitude_sobel[2]:
            raise AdvancedLaneFinderError(
                "Thresholds for magnitude Sobel x and y operators are incorrect; minimum greater than maximum [ %s ]."
                % magnitude_sobel)
        self._mag_sobel_kernel = magnitude_sobel[0]
        self._mag_sobel_thresh_min = magnitude_sobel[1]
        self._mag_sobel_thresh_max = magnitude_sobel[2]

        # thresholds (direction Sobel)
        if direction_sobel[1] > direction_sobel[2]:
            raise AdvancedLaneFinderError(
                "Thresholds for direction Sobel x and y operators are incorrect; minimum greater than maximum [ %s ]."
                % direction_sobel)
        self._dir_sobel_kernel = direction_sobel[0]
        self._dir_sobel_thresh_min = direction_sobel[1]
        self._dir_sobel_thresh_max = direction_sobel[2]

        # thresholds (S channel of HLS)
        if s_channel_thresh[0] > s_channel_thresh[1]:
            raise AdvancedLaneFinderError(
                "Thresholds for S channel of HLS image are incorrect; minimum greater than maximum [ %s ]."
                % s_channel_thresh)
        self._s_channel_thresh_min = s_channel_thresh[0]
        self._s_channel_thresh_max = s_channel_thresh[1]

        # source and destination coordinates of quadrangle vertices
        self._warp_src_vertices = warp_perspective[0]
        self._warp_dst_vertices = warp_perspective[1]
        # calculate the perspective transform matrix (and inverse)
        self._perspective_transform_matrix = \
            cv2.getPerspectiveTransform(self._warp_src_vertices, self._warp_dst_vertices)
        self._inverse_perspective_transform_matrix = \
            cv2.getPerspectiveTransform(self._warp_dst_vertices, self._warp_src_vertices)

        # params for sliding window technique used to fit polynomial
        self._sliding_window_nwindows = sliding_window_params[0]
        self._sliding_window_margin = sliding_window_params[1]
        self._sliding_window_minpix = sliding_window_params[2]

        # vertices for region of interest
        self._region_of_interest_vertices = region_of_interest_verts

        # lane lines tracking
        self._left_line = Line()
        self._right_line = Line()

        # set meters per pixel for lines
        self._left_line.meters_per_pixel_y = meters_per_pixel[0]
        self._left_line.meters_per_pixel_x = meters_per_pixel[1]

        self._right_line.meters_per_pixel_y = meters_per_pixel[0]
        self._right_line.meters_per_pixel_x = meters_per_pixel[1]

        # set maximum number of curve fit coefficients to store
        self._left_line.max_recent_xfitted = max_recent_xfitted
        self._right_line.max_recent_xfitted = max_recent_xfitted

        # linear space along Y axis
        self._ploty = np.int32(np.linspace(0, self._image_size[0]-1, self._image_size[0]))

        # lane detection failure counter
        self._lane_detection_failure_count = 0
        self._max_lane_detection_failures_before_sliding_window = lane_detection_failure_count_before_sliding_window
    
    #Grayscaling Images
    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    #HLS Scaling Images
    def HLSScale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    def Camera_Calibrate(self, image):
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # step through the list of chessboard image paths and search for chessboard corners
        for fname in self._chessboard_image_path_list:
            img = cv2.imread(fname)

            # images should have equal shape
            self._image_size == img.shape[::-1]

            gray = self.grayscale(img)

            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # if found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, self._image_size, None, None)

        if not ret:
            raise AdvancedLaneFinderError("Camera calibration has failed.")
        else:
            # initialize corresponding class instance fields
            self._calibration_matrix = mtx
            self._distortion_coefficients = dist
            self._rotation_vectors = rvecs
            self._translation_vectors = tvecs
            self._calibrated = True

            return self.get_calibration_camera_output()

    def get_calibration_camera_output(self):
        """Getter for the tuple of calibration matrix, distortion coefficients, rotation and translation vectors."""
        return (self._calibration_matrix,
                self._distortion_coefficients,
                self._rotation_vectors,
                self._translation_vectors)

    def Distortion_correction(self, image):
        return cv2.undistort(image, self._calibration_matrix, self._distortion_coefficients, None, self._calibration_matrix)

    ##Thresholds
    #HLS Saturation Image Threshold
    def HLS_Channel_S_Threshold(self, hlsimage, thresh=(170, 255)):
        s_channel = hlsimage[:,:,2]
        binary_output = np.copy(s_channel)
        binary_output[(s_channel > thresh [0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    #direction threshold for the points
    def Direction_Threshold(self, grayimage, sobel_kernel=3, thresh=(0, np.pi/2)):
        abs_sobelx = np.absolute(cv2.Sobel(grayimage, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_sobely = np.absolute(cv2.Sobel(grayimage, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        result = np.arctan2(abs_sobely, abs_sobelx)
        binary_output = np.zeros_like(result)
        binary_output[(result >= thresh[0]) & (result <= thresh[1])] = 1
        return binary_output

    #Calculating the x and Y magnitude
    def Magnitude_Threshold(self, grayimage, sobel_kernel=3, mag_thresh=(0, 255)):
        sobelx = cv2.Sobel(grayimage, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(grayimage, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output
        
    #absolute thresholds for the sobels based on X and Y
    def Absolute_Threshold(self, grayimage, orient, thresh=(0, 255)):
        if orient == 'x':
            sobel = np.abs(cv2.Sobel(grayimage, cv2.CV_64F, 1, 0))
        if orient == 'y':
            sobel = np.abs(cv2.Sobel(grayimage, cv2.CV_64F, 0, 1))
        scaled_sobel = np.uint8(255*sobel/np.max(sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] =1
        return binary_output

    #Combining all the thresholds together for the best output
    def Apply_Thresholds(self, grayimage, hlsimage):
        #getting the Thresholds details
        abs_sobel_x = self.Absolute_Threshold(grayimage, orient ='x')
        abs_sobel_y = self.Absolute_Threshold(grayimage, orient ='y')
        mag_thresh = self.Magnitude_Threshold(grayimage)
        dir_thresh = self.Direction_Threshold(grayimage)
        hls_thresh = self.HLS_Channel_S_Threshold(hlsimage)

        # combine thresholded images
        combined = np.zeros_like(dir_thresh)
        combined[((abs_sobel_x == 1) & (abs_sobel_y == 1)) | ((mag_thresh == 1) & (dir_thresh == 1)) | (hls_thresh == 1)] = 255
        return combined
    
    def Detect_lanes_Pixels(self, warped_image):
        #hyperparamaters for the sliding windows
        nwindows = 9
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
        out_img = np.dstack((warped_image, warped_image, warped_image))
        #Identifying the starting points for the lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        window_height = np.int(warped_image.shape[0]//nwindows)
        nonzero = warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base        

        for window in range(nwindows):
            win_y_low = warped_image.shape[0] - (window+1)*window_height
            win_y_high = warped_image.shape[0] - window*window_height
            
            win_x_left_low = leftx_current - margin
            win_x_left_high = leftx_current + margin
            win_x_right_low = rightx_current - margin
            win_x_right_high = rightx_current + margin
            cv2.rectangle(out_img,(win_x_left_low,win_y_low),
            (win_x_left_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_x_right_low,win_y_low),
            (win_x_right_high,win_y_high),(0,255,0), 2) 
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_left_low) &  (nonzerox < win_x_left_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_right_low) &  (nonzerox < win_x_right_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def region_of_interest(self, image):
        """Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(image)

        # defining a 3 channel or 1 channel color to fill
        # the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on an image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined
        # by "vertices" with the fill color
        cv2.fillPoly(mask, self._region_of_interest_vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def warp_perspective(self, image):
        # warp the image using OpenCV warpPerspective()
        return cv2.warpPerspective(image, self._perspective_transform_matrix, self._image_size[::-1])

    def Fit_Polynominal(self, warped_image):
        leftx, lefty, rightx, righty, out_img = self.Detect_lanes_Pixels(warped_image)

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        #Visualization
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        return out_img
    def fit_polynomial(self, image, draw=False):
        """Detect lane pixels and fit to find the lane boundary.

        Args:
            :param image: warped gray scale image
            :param draw: if true, create output image with fitted line drawn
        """
        if self._left_line.detected \
                and self._right_line.detected \
                and (self._lane_detection_failure_count < self._max_lane_detection_failures_before_sliding_window):
            return self._skip_sliding_window_fit(image=image, draw=draw)
        else:
            self._lane_detection_failure_count = 0
            self._left_line.detected = False
            self._right_line.detected = False
            return self._sliding_window_fit(image=image, draw=draw)

    def _update_lane_line(self, line, fit, allx):
        line.detected = True
        line.current_fit = fit
        line.ally = self._ploty
        line.allx = allx

    def _update_lane_lines(self, left_fit, right_fit):
        new_left_line_allx = left_fit[0]*self._ploty**2 + left_fit[1]*self._ploty + left_fit[2]
        new_right_line_allx = right_fit[0]*self._ploty**2 + right_fit[1]*self._ploty + right_fit[2]

        # check that detection was correct
        if self._left_line.detected \
           and math.fabs(new_left_line_allx[self._image_size[0]-1] - self._left_line.allx[self._image_size[0]-1]) \
               > 5 \
           and self._right_line.detected \
           and math.fabs(new_right_line_allx[self._image_size[0]-1] - self._right_line.allx[self._image_size[0]-1]) \
               > 5:
            self._lane_detection_failure_count += 1
            return

        self._update_lane_line(line=self._left_line, fit=left_fit, allx=new_left_line_allx)
        self._update_lane_line(line=self._right_line, fit=right_fit, allx=new_right_line_allx)

    def _calculate_curvature(self):
        """Calculate curvature of left and right lane lines."""
        y_eval = np.max(self._ploty)

        # fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self._left_line.ally*self._left_line.meters_per_pixel_y,
                                 self._left_line.allx*self._left_line.meters_per_pixel_x,
                                 2)
        right_fit_cr = np.polyfit(self._right_line.ally*self._right_line.meters_per_pixel_y,
                                  self._right_line.allx*self._right_line.meters_per_pixel_x,
                                  2)
        # calculate the new radius of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self._left_line.meters_per_pixel_y
                               + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self._right_line.meters_per_pixel_y
                                + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        self._left_line.radius_of_curvature = left_curverad
        self._right_line.radius_of_curvature = right_curverad

        # now radius of curvature is in meters
        return left_curverad, right_curverad

    def _calculate_bias_from_center(self):
        """Calculate bias from center of the road."""
        middle = self._image_size[1] / 2
        dist_left = (middle - self._left_line.allx[self._image_size[0]-1])
        dist_right = (self._right_line.allx[self._image_size[0]-1] - middle)

        return (dist_left - dist_right) * self._left_line.meters_per_pixel_x

    def Draw_Polygon(self, undist, warped):
        # create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([self._left_line.allx, self._left_line.ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self._right_line.allx, self._right_line.ally])))])
        pts = np.hstack((pts_left, pts_right))
        
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp,
                                      self._inverse_perspective_transform_matrix,
                                      self._image_size[::-1])
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result
    
    def Add_Text(self, image):
        l_curveradius, r_curveradius = self._calculate_curvature()
        curvaturerad = (l_curveradius + r_curveradius) / 2
        bias = self._calculate_bias_from_center()

        cv2.putText(image,
                    "Curvature Radius = %8.2f m" % curvaturerad,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    color=(255, 255, 255),
                    thickness=4)

        cv2.putText(image,
                    "Bias from Center = %8.2f m" % bias,
                    (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    color=(255, 255, 255),
                    thickness=4)
        return image

    def Pipeline(self, image):
        #Grayscaling Image
        grayimage = self.grayscale(image)

        #HLS Image
        hls = self.HLSScale(image)

        #Camera Calibrate
        mtx, dist, rvecs, tvecs = self.Camera_Calibrate(image)

        #Distortion Images
        Undistorted_Image = self.Distortion_correction(image)

        #applyting thresholds
        threshold_image = self.Apply_Thresholds(grayimage, hls)

        #Identifying the region of Interests
        region_interest_image = self.region_of_interest(threshold_image)

        #Applying the Warp perspective for the bird eye view
        warped_image = self.warp_perspective(region_interest_image)
        poly_image = self.Fit_Polynominal(warped_image)
        Draw_Poly_Image = self.Draw_Polygon(Undistorted_Image, warped_image)

        final = self.Add_Text(Draw_Poly_Image)

        return final

class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None