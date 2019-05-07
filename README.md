## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

In this project (project 2 in self-driving car Engineer Nano Degree), I was focused on creating the application pipeline to identify the lane boundaries of images and frames of videos. I have worked on finding lane in my first project using Canny and Hough Transform before. You can check the project in the following link

High-Level Overview:
---

The Target of this Project:

The High-level overview of the steps taken to achieve the target of this project:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


The Technology used:

This project is based on Python programming language, you can find the details of every library used  

The libraries used are the followings:
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import cv2
import os
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from IPython.display import display

if you need 

The Project
---
Objects used in this project

Project components:
---
Folders:
There are 5 main folders I need to highlight. For the pictures which I initially tested my application on. You can find them in the test_images and the output is stored in the output_images. As for the videos, they can be found in the input_videos and the output can be found output_videos.

If you will clone, or download the code, please make sure to delete the output images and output videos to be able to see your output of the code.

Code files for this project: 

This project is mainly composed of 2 code files. The first one is the LaneFinder.py, which list all the list of functions for the application pipeline to identify the lane. The second one is the “Project 2 - Advanced Lane Finding”, this file is the start of the project. This is where I defined the images folder, the chess board folder and the input video folder as well as the images outputs and the video outputs.

This project is using multiple objects to achieve the required target. The first one is the Line, which represents the different characteristics of every line in a lane.
The LaneFinder which identify the list of functions that will achieve the required pipeline.
 
How to test the project:
---

git clone https://github.com/hhaggan/Advanced-Lane-Finding
cd Advanced-Lane-Finding

Open the Jupyter Notebook and enjoy! Let me know your feedback.

Challenges:
---
The Input_Videos has three videos; my code is doing perfect with the project.mp4 video file. It is still not working correctly on the challenge.mp4 and the Harder_challenge.mp4. I will be working on them in my next project. 

License
---
This project is licensed under the terms of the MIT license.

## About


