import os
import logging
import math
import numpy as np
import cv2

class LaneFinder:
    def __init__(self,
                 image_size=(720, 1280),
                 chessboard_image_dir='chessboard_images',
                 absolute_sobel_x=(7, 15, 100),
                 absolute_sobel_y=(7, 15, 100),
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
                 ):
        self._image_size = image_size
        self._chessboard_image_dir = os.path.abspath(chessboard_image_dir)
        self._chessboard_image_path_list = [os.path.join(self._chessboard_image_dir, fname)
        self._calibration_matrix = None
        self._distortion_coefficients = None
        self._rotation_vectors = None
        self._translation_vectors = None
        self._calibrated = False

        self._abs_sobel_x_kernel = absolute_sobel_x[0]
        self._abs_sobel_x_thresh_min = absolute_sobel_x[1]
        self._abs_sobel_x_thresh_max = absolute_sobel_x[2]

        self._abs_sobel_y_kernel = absolute_sobel_y[0]
        self._abs_sobel_y_thresh_min = absolute_sobel_y[1]
        self._abs_sobel_y_thresh_max = absolute_sobel_y[2]

        self._mag_sobel_kernel = magnitude_sobel[0]
        self._mag_sobel_thresh_min = magnitude_sobel[1]
        self._mag_sobel_thresh_max = magnitude_sobel[2]

        self._dir_sobel_kernel = direction_sobel[0]
        self._dir_sobel_thresh_min = direction_sobel[1]
        self._dir_sobel_thresh_max = direction_sobel[2]

        self._s_channel_thresh_min = s_channel_thresh[0]
        self._s_channel_thresh_max = s_channel_thresh[1]

        self._warp_src_vertices = warp_perspective[0]
        self._warp_dst_vertices = warp_perspective[1]

        self._perspective_transform_matrix = cv2.getPerspectiveTransform(self._warp_src_vertices, self._warp_dst_vertices)
        self._inverse_perspective_transform_matrix = cv2.getPerspectiveTransform(self._warp_dst_vertices, self._warp_src_vertices)

        self._sliding_window_nwindows = sliding_window_params[0]
        self._sliding_window_margin = sliding_window_params[1]
        self._sliding_window_minpix = sliding_window_params[2]

        self._region_of_interest_vertices = region_of_interest_verts

        self._left_line = Line()
        self._right_line = Line()

        self._left_line.meters_per_pixel_y = meters_per_pixel[0]
        self._left_line.meters_per_pixel_x = meters_per_pixel[1]

        self._right_line.meters_per_pixel_y = meters_per_pixel[0]
        self._right_line.meters_per_pixel_x = meters_per_pixel[1]

        self._left_line.max_recent_xfitted = max_recent_xfitted
        self._right_line.max_recent_xfitted = max_recent_xfitted

        self._ploty = np.int32(np.linspace(0, self._image_size[0]-1, self._image_size[0]))

        self._lane_detection_failure_count = 0
        self._max_lane_detection_failures_before_sliding_window = lane_detection_failure_count_before_sliding_window

    #Step 1 Camera Calibration
    #Working with the ChessBoard
    def get_chessboard_image_list(self):
        return self._chessboard_image_path_list
    
    #Calibrating the Camera and assigning the output to the attributes
    def calibrate_camera(self):
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        for fname in self._chessboard_image_path_list:
            img = cv2.imread(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints, imageSize=img.shape[:-1], cameraMatrix=None, distCoeffs=None)

        if ret:
            self._calibration_matrix = mtx
            self._distortion_coefficients = dist
            self._rotation_vectors = rvecs
            self._translation_vectors = tvecs
            self._calibrated = True

            return self.get_calibration_camera_output()

    #getters for the output of the camera calibration
    def get_calibration_camera_output(self):
        return (self._calibration_matrix, self._distortion_coefficients, self._rotation_vectors, self._translation_vectors)
    
    #Step 2 Distortion Correction
    def distortion_correction(self, image: np.ndarray):
        return cv2.undistort(image, self._calibration_matrix, self._distortion_coefficients, None, self._calibration_matrix)

    #Step 3 Thresholding
    #Getting the direction threshold using the inverse tan
    #using sobel x and Y 
    def _dir_threshold(self, gray):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._dir_sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._dir_sobel_kernel)

        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= self._dir_sobel_thresh_min) & (absgraddir <= self._dir_sobel_thresh_max)] = 1
        return binary_output

    #Calculating the magnitude threshold
    def _mag_threshold(self, gray):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._mag_sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._mag_sobel_kernel)

        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= self._mag_sobel_thresh_min) & (gradmag <= self._mag_sobel_thresh_max)] = 1
        return binary_output

    #absolute Threshold for X or  Y
    def _abs_threshold(self, gray, orient):
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._abs_sobel_x_kernel))
            thresh_min = self._abs_sobel_x_thresh_min
            thresh_max = self._abs_sobel_x_thresh_max
        elif orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._abs_sobel_y_kernel))
            thresh_min = self._abs_sobel_y_thresh_min
            thresh_max = self._abs_sobel_y_thresh_max
        
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary_output

    #3 thresholding - Color Threshold
    def _s_channel_threshold(self, hls):
        s_channel = hls[:, :, 2]  # use S channel

        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel >= self._s_channel_thresh_min) & (s_channel <= self._s_channel_thresh_max)] = 1
        return binary_output

    #combining the thresholds together
    #magnitude, direction, absolute and HLS Thresholding
    def apply_thresholds(self, image: np.ndarray):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        abs_x_binary = self._abs_threshold(gray, orient='x')
        abs_y_binary = self._abs_threshold(gray, orient='y')
        mag_binary = self._mag_threshold(gray)
        dir_binary = self._dir_threshold(gray)
        s_channel_binary = self._s_channel_threshold(hls)

        combined = np.zeros_like(dir_binary)
        combined[((abs_x_binary == 1) & (abs_y_binary == 1))
                 | ((mag_binary == 1) & (dir_binary == 1))
                 | (s_channel_binary == 1)] = 255
        return combined

    #Step 4 Perspective Transform
    def warp_perspective(self, image):
        return cv2.warpPerspective(image, self._perspective_transform_matrix, self._image_size[::-1])

    #Step 5 Detect Lane Lines
     def _weighted_img(img1, img2, α=1.0, β=1.0, λ=0.):
        return cv2.addWeighted(img1, α, img2, β, λ)
    def region_of_interest(self, image):
        mask = np.zeros_like(image)

        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, self._region_of_interest_vertices, ignore_mask_color)

        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def _skip_sliding_window_fit(self, image, draw):
        margin = self._sliding_window_margin
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_fit = self._left_line.current_fit
        right_fit = self._right_line.current_fit
        out_img = None

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
                          & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
                           & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if draw:
            ploty = self._ploty
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            out_img = np.dstack((image, image, image))
            window_img = np.zeros_like(out_img)
            
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # generate a polygon to illustrate the search window area
            # and recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # image to draw fitted curves on
            zero_image = np.zeros_like(out_img)
            # draw fitted curves
            cv2.polylines(img=zero_image,
                          pts=np.int32(np.dstack((left_fitx, ploty))),
                          isClosed=False,
                          color=(255, 255, 0),
                          thickness=10,
                          lineType=cv2.LINE_8)
            cv2.polylines(img=zero_image,
                          pts=np.int32(np.dstack((right_fitx, ploty))),
                          isClosed=False,
                          color=(255, 255, 0),
                          thickness=10,
                          lineType=cv2.LINE_8)

            out_img = self._weighted_img(out_img, zero_image)

        # update tracked metrics of lane lines
        self._update_lane_lines(left_fit=left_fit, right_fit=right_fit)

        return out_img

    def _sliding_window_fit(self, image, draw):
        # choose the number of sliding windows
        nwindows = self._sliding_window_nwindows

        # set the width of the windows +/- margin
        margin = self._sliding_window_margin

        # set minimum number of pixels found to recenter window
        minpix = self._sliding_window_minpix

        # create an output image to draw on and visualize the result
        if draw:
            bgr_img = np.dstack((image, image, image))
            zero_img = np.zeros_like(bgr_img)
        else:
            bgr_img = None

        # take a histogram of the bottom half of the image
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)

        # find the peak of the left and right halves of the histogram
        # these will be the starting point for the left and right lines
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # set height of windows
        window_height = image.shape[0] // nwindows

        # identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # step through the windows one by one
        for window in range(nwindows):
            # identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low)
                              & (nonzeroy < win_y_high)
                              & (nonzerox >= win_xleft_low)
                              & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low)
                               & (nonzeroy < win_y_high)
                               & (nonzerox >= win_xright_low)
                               & (nonzerox < win_xright_high)).nonzero()[0]

            # draw the windows on the visualization image if required
            if draw:
                # win_y_low and win_y_high are +/- 1 to make rectangles more visible
                cv2.rectangle(zero_img,
                              (win_xleft_low, win_y_low - 1),
                              (win_xleft_high, win_y_high + 1),
                              (0, 255, 0),
                              2)
                cv2.rectangle(zero_img,
                              (win_xright_low, win_y_low - 1),
                              (win_xright_high, win_y_high + 1),
                              (0, 255, 0),
                              2)

                bgr_img[nonzeroy[good_left_inds], nonzerox[good_left_inds]] = [255, 0, 0]
                bgr_img[nonzeroy[good_right_inds], nonzerox[good_right_inds]] = [0, 0, 255]

            # append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if > minpix pixels found, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # fit a second order polynomial to each lane line
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if draw:
            ploty = self._ploty
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            cv2.polylines(img=zero_img,
                          pts=np.int32(np.dstack((left_fitx, ploty))),
                          isClosed=False,
                          color=(255, 255, 0),
                          thickness=10,
                          lineType=cv2.LINE_8)
            cv2.polylines(img=zero_img,
                          pts=np.int32(np.dstack((right_fitx, ploty))),
                          isClosed=False,
                          color=(255, 255, 0),
                          thickness=10,
                          lineType=cv2.LINE_8)
            bgr_img = self._weighted_img(bgr_img, zero_img)

        # update tracked metrics of lane lines
        self._update_lane_lines(left_fit=left_fit, right_fit=right_fit)

        return bgr_img

    def fit_polynomial(self, image, draw=False):
        if self._left_line.detected \
                and self._right_line.detected \
                and (self._lane_detection_failure_count < self._max_lane_detection_failures_before_sliding_window):
            return self._skip_sliding_window_fit(image=image, draw=draw)
        else:
            self._lane_detection_failure_count = 0
            self._left_line.detected = False
            self._right_line.detected = False
            return self._sliding_window_fit(image=image, draw=draw)
    
    #Step 6 Determine Lane Curvature
    def _calculate_bias_from_center(self):
        middle = self._image_size[1] / 2
        dist_left = (middle - self._left_line.allx[self._image_size[0]-1])
        dist_right = (self._right_line.allx[self._image_size[0]-1] - middle)

        return (dist_left - dist_right) * self._left_line.meters_per_pixel_x

    def _calculate_curvature(self):
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
    
    def add_text(self, image):
        l_curveradius, r_curveradius = self._calculate_curvature()
        curvaturerad = (l_curveradius + r_curveradius) / 2
        bias = self._calculate_bias_from_center()

        cv2.putText(image, "Curvature Radius = %8.2f m" % curvaturerad, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(255, 255, 255), thickness=4)

        cv2.putText(image, "Bias from Center = %8.2f m" % bias, (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(255, 255, 255), thickness=4)
        return image

    def draw_polygon(self, undist, warped):
        # create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self._left_line.allx, self._left_line.ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self._right_line.allx, self._right_line.ally])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix
        newwarp = cv2.warpPerspective(color_warp,
                                      self._inverse_perspective_transform_matrix,
                                      self._image_size[::-1])
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result


    #Pipeline
    def pipeline(self, image):
        if not self._calibrated:
            self.calibrate_camera()

        undist = self.distortion_correction(image=image)
        
        thresholded = self.apply_thresholds(image=undist)

        reg_of_interest = self.region_of_interest(image=thresholded)

        warped = self.warp_perspective(image=reg_of_interest)

        fit_polynomial_vis = self.fit_polynomial(image=warped, draw=(stop_on_step == 'fit_polynomial'))
        
        unwarped = self.draw_polygon(undist=undist, warped=warped)
        
        final = self.add_text(image=unwarped)

        return final

class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        # maximum size of self.recent_xfitted
        self.max_recent_xfitted = 10

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

        # meters per pixel in y dimension
        self.meters_per_pixel_y = None

        # meters per pixel in x dimension
        self.meters_per_pixel_x = None
