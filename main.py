import numpy as np
import cv2
import sys
import os
import argparse
import cv2
import imutils
import document_scanner.document_scanner as document_scanner
def Image(path):
	image = cv2.imread(path)
	ds = document_scanner.DocumentScanner()
	output = ds.Scan(image,True)
	cv2.imshow("Input",imutils.resize(image, height = 500))
	cv2.imshow("Output",output)
	cv2.waitKey(0)


if __name__=="__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--image", required = True,help = "Image Path")
	args = vars(ap.parse_args())
	Image(args['image'])
	cv2.destroyAllWindows()
