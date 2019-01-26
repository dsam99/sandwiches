import numpy as np
from PIL import Image
import glob
import os


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


def convert_directory(directory_name):
	'''
	Method to convert a directory of images into a list of lists of black and white pixel values 

	Params: pathname of the directory

	Output: A list of lists of integers 
	'''

	output_list = []

	for picture in os.listdir(directory_name):
		output_list.append(convert_image(directory_name+"/"+picture))

	return output_list





