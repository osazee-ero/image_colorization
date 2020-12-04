# This script outputs 'gray images,' 'real images,' and 'predicted images' into an output folder
# Majority of the images in our report were extracted using this script 

#%% Imports
from CustomDataGenerator import  DataGenerator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from full_colorization_network_reduced import Full_ImageColorization
import matplotlib.pyplot as plt
from helper import rgb_color_img_batches
from tensorflow.keras.layers import Input
import file_paths

# %% Model Parameters
input_shape = (256, 256, 1)
batch_size = 16
epochs = 1
img_size = 256

# %% Load Data
images_folder = file_paths.images_folder  #training set
valid_images_folder = file_paths.valid_images_folder  #validation set

training_annotations = file_paths.training_annotations
testing_annotations = file_paths.testing_annotations

output_path = file_paths.output_path

train_generator = DataGenerator(images_folder, annotations_dir=training_annotations,
                                batch_size=batch_size,
                                use_vgg=False, shuffle=True,
                                load_annotations=False,
                                img_size=img_size,
                                load_color_imgs=False,
                                num_of_instance=3,
                                use_fusion=False)

valid_generator = DataGenerator(valid_images_folder,
                                annotations_dir=testing_annotations,
                                batch_size=batch_size,
                                use_vgg=False,
                                shuffle=True,
                                img_size=img_size,
                                load_annotations=False,
                                load_color_imgs=False,
                                num_of_instance=3,
                                use_fusion=False)

# %% Load Model
full_colorization_model = Full_ImageColorization(input_shape)

# %% Load weight
recent_weights = file_paths.recent_weights
full_colorization_model.load_weights(recent_weights)

# %% define smooth l1 loss function
def smooth_L1_loss(y_true, y_pred):
    return tf.compat.v1.losses.huber_loss(y_true, y_pred)

# number of images of each set in the output
num = 10

# Iterate through images from the set (in this case we used our validation data)
for i, (grayimgs, color_imgs) in enumerate(valid_generator):
    gray_images = grayimgs
    target_images = color_imgs
    predictions = full_colorization_model.predict(gray_images)
    
    # convert to rgb
    rgb_predict_color_imgs, rgb_target_color_imgs = rgb_color_img_batches(gray_images, predictions, target_images,
                                                                          len(gray_images))
    fig = plt.figure()
    plt.clf()
    ax = fig.subplots(1, 3)
    ax[0].imshow(rgb_target_color_imgs[0, :, :, :])
    ax[0].set_title("Real_image")
    ax[0].axis("off")
    ax[1].imshow(rgb_predict_color_imgs[0, :, :, :])
    ax[1].set_title("predicted_image")
    ax[1].axis("off")

    # Convert grayscale to one channel only
    grayImg = np.squeeze(gray_images[0, :, :, :])
    ax[2].imshow(grayImg, cmap="gray")

    ax[2].imshow(gray_images[0, :, :, :])
    ax[2].set_title("grayscale_image")
    ax[2].axis("off")
    filepath = output_path + "result-train-set-" + str(i) + ".jpg"
    plt.savefig(filepath)
    plt.close(fig)

    if i == num:
        break

for i, (grayimgs, color_imgs) in enumerate(valid_generator):
    gray_images = grayimgs
    target_images = color_imgs
    predictions = full_colorization_model.predict(gray_images)
    # convert back to rgb from lab space
    rgb_predict_color_imgs, rgb_target_color_imgs = rgb_color_img_batches(gray_images, predictions, target_images,
                                                                          len(gray_images))

    fig = plt.figure()
    plt.clf()
    ax = fig.subplots(1, 3)
    ax[0].imshow(rgb_target_color_imgs[0, :, :, :])
    ax[0].set_title("Real_image")
    ax[0].axis("off")
    ax[1].imshow(rgb_predict_color_imgs[0, :, :, :])
    ax[1].set_title("predicted_image")
    ax[1].axis("off")

    # Convert grayscale to one channel only
    grayImg = np.squeeze(gray_images[0, :, :, :])
    ax[2].imshow(grayImg, cmap="gray")

    ax[2].set_title("grayscale_image")
    ax[2].axis("off")
    filepath = output_path + "result-valid-set-" + str(i) + ".jpg"
    plt.savefig(filepath)
    plt.close(fig)

    if i == num:
        break

