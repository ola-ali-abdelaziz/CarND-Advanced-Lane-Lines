#importing some useful packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
import pickle
########################################
def show_image(img,bin_img,desc=''):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(bin_img, cmap='gray')
    ax2.set_title(desc, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
def camera_calibration():
  ###  returns the calibration matrix and distortion coefficients  
    # Arrays to store object points and image points from all the Chessboard images.
    objpoints = []  # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    objp = np.zeros((9*6,3),np.float)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Use cv2.calibrateCamera() and cv2.undistort()
    # list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    img_size = (0,0)
    # loop on chesssboard images, find the findChessboardCorners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
# If the chesspint found,,, add the object point and image point
        if ret == True:
            objp = objp.astype('float32')
            corners = corners.astype('float32')
            imgpoints.append(corners)
            objpoints.append(objp)
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
# run the calibrate camera function and return the calibration matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx,dist

##call camera calibration function
mtx,dist = camera_calibration()
#print (mtx)
img = cv2.imread('camera_cal/calibration1.jpg')
##now undistort the distorted image by using undistort function and passing the calibration matrix and distortion coefficients

def undistort(img,mtx,dist):

    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
   # show_image(img,undistorted,'undistorted')
    return undistorted

undistorted = undistort(img, mtx, dist)
##save the output undistorted image
cv2.imwrite('output_images/calibration1_undist.bmp', undistorted)
#plt.imshow(undistorted)


#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(undistorted)
#ax2.set_title('Undistorted Image', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
os.listdir("camera_cal/")

#image = mpimg.imread('test_images/straight_lines1.jpg')
image = mpimg.imread('test_images/test1.jpg')
#plt.imshow(image)


# TODO: Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    show_image(img,binary_output,'binary_output')
    return binary_output

hls_binary = hls_select(image, thresh=(170, 255))
    
# Optional TODO - tune the threshold to try to match the above image!    
#hls_binary = hls_select(image, thresh=(0, 255))
cv2.imwrite('output_images/hls_binary.bmp', hls_binary*255)
# Plot the result
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(image)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(hls_binary, cmap='gray')
#ax2.set_title('hls_binary', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#image = mpimg.imread('test_images/test1.jpg')
#plt.imshow(image)

# Edit this function to create your own pipeline.
def gradients(img, s_thresh=(170, 255), sx_thresh=(50, 100)):
    img1 = np.copy(img)
    # Convert to HLS color space and separate the V channel
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #hls = cv2.cvtColor(img1, cv2.COLOR_RGB2HLS)
  #  l_channel = hls[:,:,1]
   # s_channel = hls[:,:,2]
    # Sobel x
    #sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
	# Calculate the x and y gradients
	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= 0.7) & (absgraddir <= 1.1)] = 1
    

	#combine the dir_binary img and s-binary and sobelx
    color_binary = np.zeros_like(s_binary)
    color_binary[(dir_binary == 1) & ((s_binary == 1) | (sxbinary == 1))] = 1
	
	
	
    return color_binary
    
result = gradients(image)
#show_image(image,result,'result')
cv2.imwrite('output_images/color_binary.bmp', result*255)


#cv2.imwrite('output_images/hls_binary.jpg', hls_binary)
# Plot the result
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(color_binary, cmap='gray')
#ax2.set_title('color_binary', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)





# Plot the result
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()

#ax1.imshow(image)
#ax1.set_title('Original Image', fontsize=40)

#ax2.imshow(result)
#ax2.set_title('Pipeline Result', fontsize=40)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

### Warp images to change perspective

#image = mpimg.imread('test_images/test1.jpg')
#plt.imshow(image)

def get_birdview(img):
    src = np.float32([(582,465),
          (727,465), 
          (280,690), 
          (1100,690)])

    offset = 250
    dst = np.float32([(offset,0),
          (result.shape[1]-offset,0),
          (offset,result.shape[0]),
          (result.shape[1]-offset,result.shape[0])])
    img_size = (img.shape[1], img.shape[0])

    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # save the inverse transform matrix to rectify images after processing
    M_inv = cv2.getPerspectiveTransform(dst,src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size)
          
    return warped, M, M_inv

#birdview_img, M, Minv = get_birdview(result, src, dst)
#plt.imshow(birdview_img)


# Load our image
# `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1
#img = mpimg.imread('warped_example.jpg')/255

def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
   # cv2.imwrite('output_images/histogram.bmp',histogram)
    return histogram


#show_image(binary_warped,binary_warped)
########################################
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
       # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        
        
        
        
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img

########################################################################
def blind_lane_search(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    cv2.imwrite('output_images/out_img.bmp', out_img)

    return ploty,left_fitx,right_fitx,left_fit,right_fit

# perform margin search based on a given poly and fits a poly    
def follow_up_lane_search(binary_warped,left_fit,right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return ploty,left_fitx,right_fitx,left_fit,right_fit

# display the output images with curv, offset and lanes on top    
def display_lane(undist,warped,ploty,left_fitx,right_fitx,Minv,radius,offset):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    if offset < 0:
        side = " right of center"
    else:
        side = " left of center"
    
    cv2.putText(result,"Radius = " + str(round(radius, 2)) + "(m)", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 10,thickness=5)
    cv2.putText(result,"Offset = " + str(round(abs(offset), 2)) + "m" + side, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 10,thickness=5)

    return result

# calculates the curvature radius and the offset    
def calc_rad_and_offset(ploty,left_fit,right_fit,left_fitx,right_fitx,image_size):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    
    radius = (left_curverad + right_curverad)/2.
    
    left_lane_bttom = left_fit[0]*image_size[0]**2 + left_fit[1]*image_size[0] + left_fit[2]
    right_lane_bottom = right_fit[0]*image_size[0]**2 + right_fit[1]*image_size[0] + right_fit[2]
    
    offset = (((left_lane_bttom + right_lane_bottom)/2.) - (image_size[1]/2.)) * xm_per_pix
    #print(offset, 'offset')
    
    return radius,offset
    

# global vars
left_fit = []
right_fit = []
is_first_frame = True

def pipeline(img):
    global mtx
    global dist
    undist=undistort(img,mtx,dist)

    bin_image=gradients(undist)

    #show_image(undist,undist_crop,'cropping')

    waarped,M,Minv = get_birdview(bin_image)
	
    cv2.imwrite('output_images/warped.bmp', waarped*255)
  #  show_image(bin_image*255,waarped*255,'warped')    
	
    global is_first_frame
    global right_fit
    global left_fit

    if is_first_frame == True:
        ploty,left_fitx,right_fitx,left_fit,right_fit = blind_lane_search(waarped)
        is_first_frame = False
    else:
        ploty,left_fitx,right_fitx,left_fit,right_fit = follow_up_lane_search(waarped,left_fit,right_fit)
    
    radius,offset = calc_rad_and_offset(ploty,left_fit,right_fit,left_fitx,right_fitx,waarped.shape)

    result = display_lane(undist,waarped,ploty,left_fitx,right_fitx,Minv,radius,offset)
    
    return result

white_output = 'project_video_out.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

#challenge_output = 'challenge_video_out.mp4'
#clip1 = VideoFileClip("challenge_video.mp4")
#white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
#white_clip.write_videofile(challenge_output, audio=False)

#harder_challenge_video_output = 'harder_challenge_video_output.mp4'
#clip1 = VideoFileClip("harder_challenge_video.mp4")
#white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
#white_clip.write_videofile(harder_challenge_video_output, audio=False)

########################################################################
#out_img = fit_polynomial(binary_warped)
#show_image(out_img,binary_warped)

#plt.imshow(out_img)