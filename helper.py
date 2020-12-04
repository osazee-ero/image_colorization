# This functions in this file are used in the custom data generator
# Pre-processing of the images occured here

import numpy as np
from skimage import color

#%% color batches of imgs
# Scores were used in deciding which images to return to prevent images of plain textures from being passed into training
# Furthermore, some images had a high prediction score but were undistinguishable when taken out of context
# For example people in the background (cropped images would look like random textures)
# As such image variability was also used to pass descriptive data rather than plain textures
# It was emperically found that textures like the sky or extremly blurred people in the background had variance of less than 2000
# and foreground images / in focus objects had variability of approximately 5000
def crop_images(img, boxx, scores, num_of_instance=1):
    crop_imgs = []
    bound_box = []
    if num_of_instance > len(boxx):
        num_of_instance = len(boxx)
        
    if len(boxx) > 0:
        for i,box in enumerate(boxx):
            # print(i,box)
            x1,y1,x2,y2 = np.floor(box).astype("int32")
            if len(img.shape) >= 3:
                crop_img = img[y1:y2, x1:x2, :]
            else:
                crop_img = img[y1:y2, x1:x2]
                im = np.zeros((crop_img.shape[0],crop_img.shape[1],3))
                im[:,:,0] = crop_img
                im[:,:,1] = crop_img
                im[:,:,2] = crop_img
                crop_img = im
                
            if crop_img.shape[0] == 0 or crop_img.shape[1]==0 or crop_img.var()<1000:
                continue
            else:
                if scores[i] > 0.99 and crop_img.var()>2000:
                    crop_imgs.append(crop_img) 
                    bound_box.append(np.array([x1,y1,x2,y2]))
            if i == num_of_instance:
                break
  
    return crop_imgs,bound_box 



def denorm(data):
    norm =(data -  data.mean(axis=0))/(data.std(axis=0)+1e-8)
    return norm


def restore_original_image_from_array(x, data_format='channels_last'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return x


def rgb_color_img_batches(gray_imgs, predictions, target_colorchannels, batch_size):
    # gray_imgs = restore_original_image_from_array(gray_imgs)[:] * 100.0
    gray_imgs = gray_imgs[:] * 100.0
    # gray_imgs = gray_imgs[:] * 100.0
    predictions = predictions[:] * 128.0
    # predictions = ((predictions[:]  * 2) - 1.0) * 128.0
    target_colorchannels = target_colorchannels[:] * 128.0
    # target_colorchannels = ((target_colorchannels[:] * 2) - 1.0) * 128.0
  
    predicted_color_imgs = np.concatenate((gray_imgs[:,:,:,:],predictions),axis=3)
    true_color_imgs = np.concatenate((gray_imgs[:,:,:,:],target_colorchannels),axis=3)
    
    rgb_predict_color_imgs = np.zeros_like(predicted_color_imgs)
    rgb_target_color_imgs = np.zeros_like(true_color_imgs)
    
    for i in range(batch_size):
        rgb_predict_color_imgs[i,:] = color.lab2rgb(predicted_color_imgs[i,:])
        rgb_target_color_imgs[i,:] = color.lab2rgb(true_color_imgs[i,:])
    return rgb_predict_color_imgs, rgb_target_color_imgs