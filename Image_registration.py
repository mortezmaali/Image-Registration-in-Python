import cv2
import numpy as np
  
# Open the image files.
img1_color = cv2.imread("C:/Users/Morteza/Desktop/YouTube/coding/CNN_Registration/ML_not_registered.png")  # Image to be aligned.
sh = img1_color.shape
transformed_img = np.zeros([sh[0],sh[1],sh[2]])
for j in range(0,3,2):
  
# Convert to grayscale.
    img1 = img1_color[:,:,j]
    img2 = img1_color[:,:,1]
    height, width = img2.shape
  
# Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)
  
# Find keypoints and descriptors.
# The first arg is the image, second arg is the mask
#  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
  
# Match features between the two images.
# We create a Brute Force matcher with 
# Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
  
# Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
  
# Sort matches on the basis of their Hamming distance.
    #matches.sort(key = lambda x: x.distance)
    matches = sorted(matches, key=lambda x: x.distance)
# Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
  
# Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
  
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
  
# Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
  
# Use this matrix to transform the
# first and third bands wrt the reference image.
    transformed_img[:,:,j] = cv2.warpPerspective(img1,
                    homography, (width, height))
    
transformed_img[:,:,1] = img2
transformed_img = transformed_img / transformed_img.max() #normalizes data in range 0 - 255
transformed_img = 255 * transformed_img
transformed_img = transformed_img.astype(np.uint8)

cv2.imwrite('C:/Users/Morteza/Desktop/YouTube/coding/CNN_Registration/output.jpg', transformed_img)

# concatenate image Horizontally 
Rep = np.concatenate((img1_color, transformed_img), axis=1) 

cv2.imshow('Registration Results', Rep) 

  
cv2.waitKey(0) 
cv2.destroyAllWindows() 