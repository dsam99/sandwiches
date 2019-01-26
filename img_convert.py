import numpy as np
from PIL import Image

def convert_image(jpg):
	'''
	Method to convert a jpg to a list of pixel values

	Keyword Args:
	jpg - the filename of a jpg file

	Returns:
	a list representing black and white pixel values
	'''

	im = Image.open(jpg)
	bw = im.convert("L")

	return np.array(list(bw.getdata()))

def normalize_image(image):
	'''
	Method to normalize a bw pixel values 
	'''

	image = image / 255
	return image