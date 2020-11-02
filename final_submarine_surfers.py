import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from Functions import *
from CannyEdgeDetection import *
from GrayBinaryThreshold import *
from ColorLabeler import *
from ShapeDetector import *

"""# Load Image"""
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
path = 'blue.jpg'
img=load_image(path)
plt.imshow(img)

"""# Display Color channels"""

disp_channel(img,'rgb')

"""# Kmap"""

plt.imshow(segement_kmap(img,2))

"""# Canny Edge"""

plt.imshow(auto_canny(img), cmap='gray')

"""# Thresholding"""

plt.imshow(col_thresh(image=img,ch='graythresh'),'gray')
kernel = np.ones((5,5), np.uint8) 
img_erosion = cv2.erode(img, kernel, iterations=1) 
img_dilation = cv2.dilate(img, kernel, iterations=1) 

"""#Contour Detection Detect"""

cnts,h = cv2.findContours(col_thresh(img,'graythresh'),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
plt.figure()
plt.title('Gray Binary Thresholding')
plt.imshow(cv2.drawContours(img.copy(),cnts, -1, (0, 255, 0), 2))
cnts1,h = cv2.findContours(auto_canny(img.copy()),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
plt.figure()
plt.title('Canny Edge Thresholding')
plt.imshow(cv2.drawContours(img.copy(),cnts1, -1, (0, 255, 0), 2))

color=(input('Please enter color of the door '))
h=float(input('Please enter height of the door '))
w=float(input('Please enter width of the door '))

"""# Centre Detection on Canny Edge Thresholded"""

selected_contor_canny= findshapeincanny(img.copy(),w,h)

"""## Centre detection on Gray Binary **Thresholded**"""

selected_contor_gray= findshapeingray(img.copy(),w,h)



