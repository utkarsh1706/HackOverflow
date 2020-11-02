import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from Functions import *
def findshapeingray(image1,w,h):
	image=image1.copy()
	resized = imutils.resize(image, width=300)
	newimage=np.copy(resized)
	#plt.show()
	ratio = image.shape[0] / float(resized.shape[0])
	wc=w/ratio
	hc=h/ratio
	expected_area=wc*hc
	print(expected_area)
	cnts,h = cv2.findContours(col_thresh(resized,'graythresh'),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	print('Number of Countours detected: ',len(cnts))
	selected_contor=[]
	for c in cnts:
		M = cv2.moments(c)	# compute the center of the contour, then detect the name of the
		cX = int((M["m10"] / (M["m00"]+0.00001)) * ratio)	# shape using only the contour
		cY = int((M["m01"] / (M["m00"]+0.00001)) * ratio)
		condn,color,shape = constraint(c,newimage)
		text = "{} {}".format(color, shape)
		area=cv2.contourArea(c)
		print(area)
		if((shape=='rectangle' or shape=='square') and area>=expected_area/10):
			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the nam;e of the shape on the image
			c = c.astype("float")
			c *= ratio
			c = c.astype("int")
			image=image1.copy()
			cv2.drawContours(image, [c], 0, (0, 255, 0), 2)
			cv2.putText(image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,	0.5, (255, 255, 255), 2)
			cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
			ares=' '
			plt.figure()
			if (abs(area-expected_area)/expected_area)<=0.6:
				ares=' area is: ' + str(area)
			plt.title('Shape is: '+shape+', Color is: ' + color + ares)
			plt.imshow(image)	# show the output image
			selected_contor.append(c)
	return selected_contor

