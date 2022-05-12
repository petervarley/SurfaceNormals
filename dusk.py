#---------------------------------------------------------------------------
# See pp.README.txt for instructions.
#---------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import pp

#---------------------------------------------------------------------------

def dusk(source):
	original = cv2.imread(source)
	night = model.forward(original)
	return original/2 + night/2

#---------------------------------------------------------------------------

if __name__ == '__main__':
	outpath = 'dusk'
	model = pp.Pix2PixModel('day2night')
	os.makedirs(outpath,exist_ok=True)

	if os.path.isdir(sys.argv[1]):
		for fname in os.listdir(sys.argv[1]):
			if pp.is_image_file(fname):
				image_path = os.path.join(sys.argv[1], fname)
				image = dusk(image_path)
				cv2.imwrite(os.path.join(outpath,fname),image)

	elif pp.is_image_file(sys.argv[1]):
		image_path = sys.argv[1]
		image = dusk(image_path)
		cv2.imwrite(os.path.join(outpath,os.path.basename(image_path)),image)

#---------------------------------------------------------------------------
