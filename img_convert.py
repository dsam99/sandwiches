import numpy as np
from PIL import Image
import glob
import os
import pickle


def convert_image(jpg):
	'''
	Method to convert a jpg to a list of pixel values

	Keyword Args:
	jpg - the filename of a jpg file

	Returns:
	a list representing black and white pixel values
	'''
	im = Image.open(jpg)

	# Resizes image to be 512 by 384
	im = im.resize((512,384))

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
		output_list.append(normalize_image(convert_image(directory_name+"/"+picture)))

	return output_list


def resize_image(image):
	'''
	Method to resize given image to a uniform width and height

	Params: pathname to an image

	Returns: Nothing 
	'''
	pass 

def store_data(directory_name):
	'''
	Method to serialize and store converted image data

	Params: A pathname to a directory containing other directories which contain images

	Output: Nothing
	'''
	with open('image_values.pickle',"wb") as f:
		for inner_directory_name in os.listdir(directory_name):
			pickle.dump(convert_directory(directory_name + "/" + inner_directory_name),f)
			break

