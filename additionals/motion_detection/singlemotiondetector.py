# import the necessary packages
import numpy as np
import imutils
import cv2

class SingleMotionDetector:
	def __init__(self, weight=0.5):
		self.weight = weight

		self.background = None

	def update(self, image):

		if self.background is None:
			self.background = image.copy().astype("float")
			return

		cv2.accumulateWeighted(image, self.background, self.weight)

	def detect(self, image, tVal=25):
		diff = cv2.absdiff(self.background.astype("uint8"), image)
		threshhold = cv2.threshold(diff, tVal, 255, cv2.THRESH_BINARY)[1]


		threshhold = cv2.erode(threshhold, None, iterations=2)
		threshhold = cv2.dilate(threshhold, None, iterations=2)

		cnts = cv2.findContours(threshhold.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)

		if len(cnts) == 0:
			return None


		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			(minX, minY) = (min(minX, x), min(minY, y))
			(maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

		return (threshhold, (minX, minY, maxX, maxY))