# Helper code / script to clean up data
# This script is intended to be run after bounding boxes have been saved
# The bounding box script used was the same as the original authors
# This script will delete any image that does not have bounding boxes
# This script will delete any image that is grayscale
# Furthermore this script will split the dataset into 90/10 training / validation

### Note: images and their associated bounding boxes are named the same
### For example:      img = example.jpg      bbox = example.npz

# Imports
import os
from os.path import isfile, join
import shutil
import io

# Paths to data
images_folder = r'training_data/training_images'
tests_folder = r'training_data/testing_images'

training_annotations = r'training_data/training_annotations'
testing_annotations = r'training_data/testing_annotations'

# Remove images with no bounding box
# variables c and d are indicies used to iterate over data
# variable v is used to specify the bounding box file

c = 0
d = 0
for f in os.listdir(images_folder):
    v = f[0:-4]
    v += ".npz"
    c += 1
    if not isfile(join(training_annotations, v)):
        c -= 1
        d += 1
        print("deleting " + f)
        os.remove(join(images_folder, f))
print("deleted ", d, " images, ", c, " images left.")

# Split training and validation images
# 90/10 split

c1 = 0
c = c / 10
for f in os.listdir(images_folder):
    c1 += 1
    v = f[0:-4]
    v += ".npz"
    shutil.move(join(images_folder, f), join(tests_folder, f))
    shutil.move(join(training_annotations, v), join(testing_annotations, v))
    if c1 >= c:
        print("moved ", c1, " images to testing directory")
        break

# delete greyscale images based on the number of channels
# Assumes grayscale images are only 1 channel and not 3 channels
d1 = 0
for f in os.listdir(images_folder):
    image = io.imread(join(images_folder, f))
    if len(image.shape) < 3:
        os.remove(join(images_folder, f))
        v = f[0:-4]
        v += ".npz"
        os.remove(join(training_annotations, v))
        d1 += 1
for f in os.listdir(tests_folder):
    image = io.imread(join(tests_folder, f))
    if len(image.shape) < 3:
        os.remove(join(tests_folder, f))
        v = f[0:-4]
        v += ".npz"
        os.remove(join(testing_annotations, v))
        d1 += 1
print("removed ", d1, " greyscale images")
