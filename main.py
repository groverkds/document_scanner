import cv2
import argparse
import document_scanner.document_scanner as document_scanner
def Image(path):

	image = cv2.imread(path)
	ds = document_scanner.DocumentScanner()
	output = ds.Scan(image,perform_threshold=True,perform_auto_rotate=True)
	
	tc = document_scanner.TextConverter()
	text = tc.ConvertImageToText(output)
	print(text)
	
	cv2.imshow("Input",document_scanner.resize_to_height(image, 800))
	
	#cv2.imshow("Output",output)
	cv2.imshow("Output",document_scanner.resize_to_height(output, 800 if output.shape[0]>800 else output.shape[0]))
	
	cv2.waitKey(0)


if __name__=="__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--image", required = True,help = "Image Path")
	args = vars(ap.parse_args())
	Image(args['image'])
	cv2.destroyAllWindows()
