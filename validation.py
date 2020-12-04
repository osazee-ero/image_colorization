# This script was used to get inference time as well as metrics such as PSNR and SSIM as used in the paper
#%% Imports

import numpy as np
from skimage import color
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from CustomDataGenerator import DataGenerator

import tensorflow as tf
from tensorflow import keras
from full_colorization_network_reduced import Full_ImageColorization
import matplotlib.pyplot as plt
from helper import rgb_color_img_batches
from tensorflow.keras.layers import Input
import cv2

import time
from statistics import mean
import file_paths

# %% Model Parameters
input_shape = (128, 128, 1)
batch_size = 16
epochs = 1
img_size = 128

# %% Load data
images_folder = file_paths.images_folder  #training set
valid_images_folder = file_paths.valid_images_folder  #validation set

training_annotations = file_paths.training_annotations
testing_annotations = file_paths.testing_annotations


# Load model and it's weights to test PSNR and SSIM metrics
full_colorization_model = Full_ImageColorization(input_shape)
recent_weights = file_paths.recent_weights
full_colorization_model.load_weights(recent_weights)
# %% calculate values

valid_generator  = DataGenerator(valid_images_folder,
                                 annotations_dir=testing_annotations,
                                 batch_size=batch_size, 
                                 use_vgg=False, 
                                 shuffle=True, 
                                 img_size=img_size, 
                                 load_annotations=False, 
                                 load_color_imgs=False,
                                 num_of_instance=3,
                                 use_fusion=False)

num = 100
psnrArr = []
ssimArr = []
tic = time.perf_counter()
for i,(grayimgs,color_imgs) in enumerate(valid_generator):

    print(grayimgs.shape, color_imgs.shape)

    gray_images = grayimgs
    target_images = color_imgs
    predictions = full_colorization_model.predict(gray_images)
    # convert to rgb
    rgb_predict_color_imgs, rgb_target_color_imgs = rgb_color_img_batches(gray_images, predictions, target_images,
                                                                          len(gray_images))

    real = np.uint8(255*rgb_target_color_imgs[0, :, :, :])

    pred = np.uint8(255*grayimgs[0, :, :, :])
    pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2RGB)

    psnrVal = psnr(real, pred)
    ssimVal = ssim(real, pred, multichannel=True)

    psnrArr.append(psnrVal)
    ssimArr.append(ssimVal)

    if i == num:
        break

toc = time.perf_counter()
print("Time Taken: ", toc-tic)
print("Avg PSNR: ", mean(psnrArr))
print("Avg SSIM: ", mean(ssimArr))
