"""
Usage: python label_images.py <image directory> <label file>
Labels are appended to the file, so the same file can be used repeatedly.
"""

import csv
from PIL import Image
from os import listdir
from os.path import abspath, isfile, join
import sys

try:
    input = raw_input
except NameError:
    pass

path = sys.argv[1]
save = sys.argv[2]

images = [f for f in listdir(path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".JPG") or f.endswith(".JPEG")]
images = images[0:20]

dir = abspath(path).split("\\")[-1]

with open(save, "a+") as f:
    writer = csv.writer(f, delimiter=",", quotechar="\"")

    for i in images:
        with Image.open(join(path, i)) as image:
            image.show()

        label = input()
        writer.writerow([join(dir, i), label])
