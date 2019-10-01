# import the necessary packages
import numpy as np
import imutils
import cv2
import os 
 
# load the image image, convert it to grayscale, and detect edges
template_original = cv2.imread("image/template/gold-bar-template-2.png")
template = cv2.cvtColor(template_original, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imwrite("template.jpg",template)

# load the image, convert it to grayscale, and initialize the
# bookkeeping variable to keep track of the matched region
# image = cv2.imread("image/false-gold-bar/1.jpg")
image = cv2.imread("image/true-gold-bar/success-2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None
# loop over the scales of the image
for scale in np.linspace(0.5, 1.0, 40)[::-1]:
	# resize the image according to the scale, and keep track
	# of the ratio of the resizing
	resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
	r = gray.shape[1] / float(resized.shape[1])
	# if the resized image is smaller than the template, then break
	# from the loop
	if resized.shape[0] < tH or resized.shape[1] < tW:
		break
	# detect edges in the resized, grayscale image and apply template
	# matching to find the template in the image
	edged = cv2.Canny(resized, 50, 200)
	result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
	# 
	if found is None:
		found = (maxVal, maxLoc, r)
	elif maxVal > found[0]:
		found = (maxVal, maxLoc, r)

# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

print(maxVal)

# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)
cv2.imshow("Image", image)
# cv2.imwrite("eus.jpg",image)
cv2.waitKey(0)