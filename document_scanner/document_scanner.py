import numpy as np
import cv2
import sys
import os
import imutils
from PIL import Image
import pytesseract

class DocumentScanner:
	def __init__(self):
		pass

	def Scan(self,image,perform_threshold=False):
		
		original = image.copy()
		
		try:
			ratio = image.shape[0]/500
			image = imutils.resize(image, height = 500)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (5,5), 0)
			edge_map = cv2.Canny(gray, 75,200)
			
			page_contour,page_corners = self.find_page(edge_map)
			
			page_corners = np.array((ratio*np.array(page_corners,dtype="float32")),dtype="int32")

			page = self.straighten_and_crop(page_corners,original)
			page = self.threshold(page,perform_threshold)
		
		except Exception as e:
			print(e)
			page = original
		
		return page


	def find_page(self,edge_map):
		
		contours = cv2.findContours(edge_map.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)
		
		for c in contours:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		
			if len(approx) == 4:
				page_corners = approx
				page_contour = c
				break
		
		return page_contour,page_corners

	def straighten_and_crop(self,page_corners,image):
		
		top_left,top_right,bottom_right,bottom_left = page_corners
		
		new_height = np.sqrt(max((top_left[0,0]-bottom_left[0,0])**2+(top_left[0,1]-bottom_left[0,1])**2,(top_right[0,0]-bottom_right[0,0])**2+(top_right[0,1]-bottom_right[0,1])**2))
		new_height = int(new_height)
		
		new_width = np.sqrt(max((top_left[0,0]-top_right[0,0])**2+(top_left[0,1]-top_right[0,1])**2,(bottom_left[0,0]-bottom_right[0,0])**2+(bottom_left[0,1]-bottom_right[0,1])**2))
		new_width = int(new_width)

		page_dim = np.array([[0, 0],[new_width - 1, 0],[new_width - 1, new_height - 1],[0, new_height - 1]], dtype = "float32")
		
		transform_vector = cv2.getPerspectiveTransform(np.array(page_corners,dtype="float32"), page_dim)
		page = cv2.warpPerspective(image, transform_vector, (new_width, new_height))
		
		page = imutils.rotate_bound(cv2.flip(page, 0),+90)
		
		return page

	def threshold(self,page,perform_threshold):

		if perform_threshold:
			page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
			page = cv2.adaptiveThreshold(page,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,page.shape[0]//10+1,page.shape[0]//100+1)		
		
		return page

class TextConverter:
	def __init__(self):
		pass

	def ConvertImageToText(self,image):
		return pytesseract.image_to_string(Image.fromarray(image), lang='eng')