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
	# Returns corners as top-left, top-right, bottom-right, bottom-left: sorted clockwise
	# around their centre, starting from the corner closest to the image's top-left
	corners = corners.reshape(4, 2)
	centre = corners.mean(axis=0)
	corners = corners[np.argsort(np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0]))]
	return np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0)

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

# Page detection works on a copy of the image resized to this height
DETECTION_HEIGHT = 500

# A page outline must cover at least this fraction of the image
MIN_PAGE_AREA_FRACTION = 0.05

def edge_maps(gray):
	# Yields edge maps to search for the page outline, from the most precise to the most forgiving
	blurred = cv2.GaussianBlur(gray, (5,5), 0)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	edges = cv2.Canny(blurred, 75, 200)
	yield edges
	# Close small gaps in the outline
	yield cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	# Lower thresholds based on the image's median brightness, for low-contrast backgrounds
	median = np.median(blurred)
	low_contrast = cv2.Canny(blurred, int(max(0, 0.67 * median) * 0.25), int(min(255, 1.33 * median) * 0.5))
	yield cv2.morphologyEx(low_contrast, cv2.MORPH_CLOSE, kernel)
	# Bright page against a darker background
	_, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	yield cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)))

def line_intersection(line1, line2):
	(x1, y1), (x2, y2) = line1
	(x3, y3), (x4, y4) = line2
	denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
	if abs(denominator) < 1e-6:
		return None
	a = x1 * y2 - y1 * x2
	b = x3 * y4 - y3 * x4
	return np.array([(a * (x3 - x4) - (x1 - x2) * b) / denominator, (a * (y3 - y4) - (y1 - y2) * b) / denominator])

def quad_from_lines(lines):
	# Intersects four lines, given in order around the page, into four corners
	corners = [line_intersection(lines[i - 1], lines[i]) for i in range(4)]
	if any(c is None for c in corners):
		return None
	return np.array(corners, dtype="float32")

def approximate_quad(contour):
	# Returns four corners approximating the contour, or None if it is not four-sided
	peri = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
	if len(approx) != 4:
		return None
	return approx.reshape(4, 2).astype("float32")

def longest_sides_quad(contour):
	# Corners cut off by a fold, or sides broken up by a finger, add short sides to the
	# outline; intersecting its four longest sides recovers the page corners
	approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True).reshape(-1, 2).astype("float32")
	if len(approx) < 4:
		return None
	sides = [(approx[i], approx[(i + 1) % len(approx)]) for i in range(len(approx))]
	longest = sorted(np.argsort([-np.linalg.norm(b - a) for a, b in sides])[:4])
	return quad_from_lines([sides[i] for i in longest])

def side_support(quad, outline_distance):
	# Fraction of the least supported side that lies on the outline; outline_distance is the
	# distance transform of the outline, giving each pixel's distance to the nearest outline point
	h, w = outline_distance.shape
	support = 1.0
	for i in range(4):
		samples = np.linspace(quad[i], quad[(i + 1) % 4], 50)
		x = np.clip(samples[:, 0].round().astype(int), 0, w - 1)
		y = np.clip(samples[:, 1].round().astype(int), 0, h - 1)
		inside = (samples[:, 0] >= 0) & (samples[:, 0] < w) & (samples[:, 1] >= 0) & (samples[:, 1] < h)
		support = min(support, np.mean(inside & (outline_distance[y, x] <= 3)))
	return support

# Every side of a page outline must lie on the detected outline for at least this fraction of its length
MIN_SIDE_SUPPORT = 0.6

def touches_border(quad, shape, margin=3):
	# True if a side of the quad runs along the image border (an outline of the image itself, not a page)
	h, w = shape
	for a, b in zip(quad, np.roll(quad, -1, axis=0)):
		for axis, limit in ((0, w - 1), (1, h - 1)):
			if max(a[axis], b[axis]) <= margin or min(a[axis], b[axis]) >= limit - margin:
				return True
	return False

class DocumentScanner:
	def __init__(self):
		pass

	def Scan(self,image,perform_threshold=False,perform_auto_rotate=False):
		
		original = image.copy()
		
		try:
			page_corners = self.detect_page(image)
			page = self.straighten_and_crop(page_corners,original)
			if perform_auto_rotate:
				page = auto_rotate(page)
			page = self.threshold(page,perform_threshold)
		
		except PageNotFoundError as e:
			print("Warning: %s; returning the original image" % e, file=sys.stderr)
			page = original
		
		return page


	def detect_page(self,image):
		# Returns the page corners in image coordinates, ordered top-left, top-right, bottom-right, bottom-left
		ratio = image.shape[0] / DETECTION_HEIGHT
		gray = cv2.cvtColor(resize_to_height(image, DETECTION_HEIGHT), cv2.COLOR_BGR2GRAY)
		for edge_map in edge_maps(gray):
			try:
				return self.find_page(edge_map) * ratio
			except PageNotFoundError:
				pass
		raise PageNotFoundError("no page outline found")

	def find_page(self,edge_map):
		
		contours, _ = cv2.findContours(edge_map.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)
		min_area = MIN_PAGE_AREA_FRACTION * edge_map.shape[0] * edge_map.shape[1]
		
		for c in contours:
			if cv2.contourArea(c) < min_area:
				break
			outline = np.full(edge_map.shape, 255, np.uint8)
			cv2.drawContours(outline, [c], -1, 0, 1)
			outline_distance = cv2.distanceTransform(outline, cv2.DIST_L2, 3)
			for quad in (approximate_quad(cv2.convexHull(c)), longest_sides_quad(c)):
				if quad is None or not cv2.isContourConvex(quad.astype("int32")) or cv2.contourArea(quad) < min_area:
					continue
				if touches_border(quad, edge_map.shape):
					continue
				if side_support(quad, outline_distance) >= MIN_SIDE_SUPPORT:
					return order_corners(quad)
		
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