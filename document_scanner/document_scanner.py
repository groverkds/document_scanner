import sys
import numpy as np
import cv2
from PIL import Image
import pytesseract
from pytesseract import Output

def resize_to_height(image, height):
	ratio = height / image.shape[0]
	return cv2.resize(image, (int(image.shape[1] * ratio), height), interpolation=cv2.INTER_AREA)

def adaptive_threshold(gray):
	# blockSize must be odd and > 1
	block_size = max(3, (gray.shape[0]//10) | 1)
	return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,gray.shape[0]//100+1)

def order_corners(corners):
	# Returns corners as top-left, top-right, bottom-right, bottom-left
	corners = corners.reshape(4, 2)
	s = corners.sum(axis=1)
	d = np.diff(corners, axis=1).ravel()
	return np.array([corners[np.argmin(s)], corners[np.argmin(d)], corners[np.argmax(s)], corners[np.argmax(d)]])

# Clockwise rotations, keyed by the "rotate" value from Tesseract's orientation detection
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

# Below this orientation confidence the detected rotation is ignored (images without
# text score around 0.5, the sample receipt about 2-4, text-heavy pages over 10)
MIN_ORIENTATION_CONFIDENCE = 1.0

def auto_rotate(page):
	# Rotates the page so its text is upright, using Tesseract's orientation detection
	gray = page if page.ndim == 2 else adaptive_threshold(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY))
	try:
		osd = pytesseract.image_to_osd(gray, output_type=Output.DICT)
	except pytesseract.TesseractError:
		# Raised when the page has too little text to detect orientation
		return page
	if osd["orientation_conf"] < MIN_ORIENTATION_CONFIDENCE or osd["rotate"] not in ROTATIONS:
		return page
	return cv2.rotate(page, ROTATIONS[osd["rotate"]])

class PageNotFoundError(Exception):
	pass

# A page outline must cover at least this fraction of the image
MIN_PAGE_AREA_FRACTION = 0.1

class DocumentScanner:
	def __init__(self):
		pass

	def Scan(self,image,perform_threshold=False,perform_auto_rotate=False):
		
		original = image.copy()
		
		try:
			ratio = image.shape[0]/500
			image = resize_to_height(image, 500)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (5,5), 0)
			edge_map = cv2.Canny(gray, 75,200)
			
			page_corners = self.find_page(edge_map)
			
			page_corners = np.array((ratio*np.array(page_corners,dtype="float32")),dtype="int32")

			page = self.straighten_and_crop(page_corners,original)
			if perform_auto_rotate:
				page = auto_rotate(page)
			page = self.threshold(page,perform_threshold)
		
		except PageNotFoundError as e:
			print("Warning: %s; returning the original image" % e, file=sys.stderr)
			page = original
		
		return page


	def find_page(self,edge_map):
		
		contours, _ = cv2.findContours(edge_map.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)
		min_area = MIN_PAGE_AREA_FRACTION * edge_map.shape[0] * edge_map.shape[1]
		
		for c in contours:
			if cv2.contourArea(c) < min_area:
				break
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		
			if len(approx) == 4 and cv2.contourArea(approx) >= min_area:
				return order_corners(approx)
		
		raise PageNotFoundError("no page outline found")

	def straighten_and_crop(self,page_corners,image):
		
		top_left,top_right,bottom_right,bottom_left = page_corners
		
		new_height = np.sqrt(max((top_left[0]-bottom_left[0])**2+(top_left[1]-bottom_left[1])**2,(top_right[0]-bottom_right[0])**2+(top_right[1]-bottom_right[1])**2))
		new_height = int(new_height)
		
		new_width = np.sqrt(max((top_left[0]-top_right[0])**2+(top_left[1]-top_right[1])**2,(bottom_left[0]-bottom_right[0])**2+(bottom_left[1]-bottom_right[1])**2))
		new_width = int(new_width)

		page_dim = np.array([[0, 0],[new_width - 1, 0],[new_width - 1, new_height - 1],[0, new_height - 1]], dtype = "float32")
		
		transform_vector = cv2.getPerspectiveTransform(np.array(page_corners,dtype="float32"), page_dim)
		page = cv2.warpPerspective(image, transform_vector, (new_width, new_height))
		
		return page

	def threshold(self,page,perform_threshold):

		if perform_threshold:
			page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
			page = adaptive_threshold(page)
		
		return page

class TextConverter:
	def __init__(self):
		pass

	def ConvertImageToText(self,image):
		# Grayscale input (e.g. an already thresholded scan) is used as is
		if image.ndim == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = adaptive_threshold(image)
		return pytesseract.image_to_string(Image.fromarray(image), lang='eng')