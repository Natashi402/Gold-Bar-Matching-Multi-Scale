# Import the necessary packages
import numpy as np
import imutils
import cv2
import os 
 
# Load the image image, convert it to grayscale, and detect edges
template_original = cv2.imread("image/template/gold-bar-template-2.png")
template = cv2.cvtColor(template_original, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
# Looking at the true-gold-bar folder to find those images
path = ['image/true-gold-bar','image/false-gold-bar']
# path = ['image/false-gold-bar']

for p in path:
	print(p)
	files = []
	# r=root, d=directories, f = files
	for r, d, f in os.walk(p):
		for file in f:
			if '.jpg' in file:
				files.append(file)
	# Perform the algorithm in each image
	for f in files:
		# Load the image, convert it to grayscale, and initialize the
		# bookkeeping variable to keep track of the matched region
		image = cv2.imread(p+"/"+f)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		found = None
		# Loop over the scales of the image
		for scale in np.linspace(0.1, 1.0, 70)[::-1]:
			# Resize the image according to the scale, and keep track
			# of the ratio of the resizing
			width = int(gray.shape[1] * scale)
			height = int(gray.shape[0] * scale)
			dim = (width, height)
			# resize image
			resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
			r = gray.shape[1] / float(resized.shape[1])
			# If the resized image is smaller than the template, then break
			# from the loop
			if resized.shape[0] < tH or resized.shape[1] < tW:
				break
			# Detect edges in the resized, grayscale image and apply template
			# matching to find the template in the image
			edged = cv2.Canny(resized, 50, 200)
			result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
			
			# 
			if found is None:
				found = (maxVal, maxLoc, r)
			elif maxVal > found[0]:
				found = (maxVal, maxLoc, r)

		# Unpack the bookkeeping variable and compute the (x, y) coordinates
		# of the bounding box based on the resized ratio
		if found is not None:
			(_, maxLoc, r) = found
			(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
			(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
			print(maxVal/10000)
			# Draw a bounding box around the detected result and write the image to the result folder
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.imwrite("image/result-failed/"+f,image)