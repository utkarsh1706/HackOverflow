import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from ColorLabeler import *
from ShapeDetector import *
def disp_channel(image,space):
    if(space=='rgb'):
        r = image[:,:,0]
        g = image[:,:,1]
        b = image[:,:,2]
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
        ax1.set_title('Red')
        ax1.imshow(r, cmap='gray')
        ax2.set_title('Green')
        ax2.imshow(g, cmap='gray')
        ax3.set_title('Blue')
        ax3.imshow(b, cmap='gray')
    elif(space=='hsv'):
        # Convert from RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # HSV channels
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
        ax1.set_title('Hue')  #Hue threshold produces better distinction between pink and other colors
        ax1.imshow(h, cmap='gray')
        ax2.set_title('Saturation')
        ax2.imshow(s, cmap='gray')
        ax3.set_title('Value')
        ax3.imshow(v, cmap='gray')


"""# Kmap"""

def segement_kmap(image,k):
    pixel_vals = image.reshape((-1,3)) 
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    #print(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))
    labels_reshape = labels.reshape(image.shape[0], image.shape[1])
    return segmented_image


"""# Canny Edge"""

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 30, 200)
    plt.figure()
    return edged
def auto_canny(image, sigma=0.33):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  v = np.median(blurred)	# compute the median of the single channel pixel intensities
  lower = int(max(0, (1.0 - sigma) * v))	# apply automatic Canny edge detection using the computed median
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(blurred, lower, upper)
  plt.figure()
  # return the edged image
  return edged  


"""# Thresholding"""

def col_thresh(image,ch,lower=[0],upper=[120]):
  if(ch=='hsv'):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    th = cv2.inrange(hsv,lower,upper)
    plt.figure()
    return th
  elif(ch=='graythresh'):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret3,th = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return th

def returnarea(cnts):
  areas=[]
  for c in cnts:
    areas.append(cv2.contourArea(c))
  maxi=max(areas)
  return maxi
def constraint(c,img):
  shape = detect(c)
  blurred = cv2.GaussianBlur(img, (5, 5), 0)
  gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
  lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
  cl = ColorLabeler()
  color = cl.label(lab, c)
  if(color=='red' and (shape=='rectangle' or shape=='square')):
    return True,color,shape
  else:
    return False,color,shape