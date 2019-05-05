#defining the list of libraries 
import os 
import numpy as numpy
import matplitlib.image as mpimg
import matplotlib.pyplot as pyplot
import math
import cv2
'''that's my great source of learning https://github.com/ser94mor/advanced-lane-finding '''

#Constants
image_size = (720, 1280)
chessboard_image_dir = "chessboard_images"
src = np.float32([(532, 496), (756, 496), (288, 664), (1016, 664)])
dst =  np.float32([(288, 366), (1016, 366), (288, 664), (1016, 664)])


class AdvanceLaneFinding():

    #mainly initializing the Lines Class

    #Perspective Transform
    def Perspective_Transform(self, image):

    #Grayscaling the images
    def grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def Distortion_correction(self, image):
        return cv2.undistort(image, self._calibration_matrix, self._distortion_coefficients, None, self._calibration_matrix)

    def HLSScale(image):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    #Color & Gradient Threshold
    def Color_Gradient_Threshold(self, image):
        
    #Camera Calibration
    def Camera_Calibration(self, image):

    ##Thresholds
    #HLS Saturation Image Threshold
    def HLS_Channel_S_Threshold(self, hlsimage, thresh=(170, 255):
        s_image = hlsimage[:,:,2]
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
    def Absolute_Threshold(self, grayimage, orient):
        if orient == 'x':
            sobel = np.abs(cv2.Sobel(grayimage, cv2.CV_64F, 1, 0))
        if orient == 'y':
            sobel = np.abs(cv2.Sobel(grayimage, cv2.CV_64F, 0, 1))
        scaled_sobel = np.uint8(255*sobel/np.max(sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] =1
        return binary_output

    #Combining all the thresholds together for the best output
    def Apply_Thresholds(self, image):
        #applyign the grayscale and the HLS 
        grayimage = grayscale(image)
        hlsimage = self.HLSscale(image)

        #getting the Thresholds details
        abs_sobel_x = self.Absolute_Threshold(grayimage, orient ='x')
        abs_sobel_y = self.Absolute_Threshold(grayimage, orient ='y')
        mag_thresh = self.Magnitude_Threshold(grayimage)
        dir_thresh = self.direction_threshold(grayimage)
        hls_thresh = self.HLS_Channel_S_Threshold(hlsimage)

        # combine thresholded images
        combined = np.zeros_like(dir_binary)
        combined[((abs_sobel_x == 1) & (abs_sobel_y == 1))
                 | ((mag_thresh == 1) & (dir_thresh == 1))
                 | (hls_thresh == 1)] = 255
        return combined

    #window
    #Detect Lane Lines
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
    
    #Determine Lane Curvatures
    def Lane_Curvatures(self, image):

    #function to identify the weighted functions
    def Weighted_Image(image1, image2, α=1.0, β=1.0, λ=0.):
        return cv2.addWeighted(image1, image2, α, β, λ)

    def Warp_Image(self, image):
        return cv2.warpPerspective(image, self._perspective_transform_matrix, self._image_size[::-1])

    def add_text(self, image):
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

    #function to be called that has all the required steps to achieve 
    #the advanced Lane Findings
    def pipeline(self, image):
        #Calibrate the Camera
        calibrated_image = self.Camera_Calibration(image)

        #undistort the Images
        undistored_image = self.Distortion_correction(calibratedimage)

        #applying Thresholds
        threshold_image = self.Apply_Thresholds(undistored_image)

        #Identifying the region of Interests
        region_interest_image = self.Define_Region_Interest(threshold_image)

        #Applying the Warp perspective for the bird eye view
        warped_image = self.Warp_Image(region_interest_image)
        poly_image = self.Fit_Polynominal(warped_image)

        #Lines Curvatures


# Define a class to receive the characteristics of each line detection
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