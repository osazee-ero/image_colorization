# Script used to run our instance image colorization network
# Note: this script greatly resembles the full image colorization training except for a few parameters
# We thought it would be best to keep these separate as we experimented with various hyper parameters for each network
# %% Imports
from CustomDataGenerator import DataGenerator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from full_colorization_network_reduced import Full_ImageColorization
import matplotlib.pyplot as plt
from helper import rgb_color_img_batches
from tensorflow.keras.layers import Input
import file_paths

# %% Model Parameters
input_shape = (128, 128, 1)
batch_size = 8
epochs = 1
img_size = 128

# %% Load Data
images_folder = file_paths.images_folder  # training set
valid_images_folder = file_paths.valid_images_folder  # validation set

training_annotations = file_paths.training_annotations
testing_annotations = file_paths.testing_annotations

train_generator = DataGenerator(images_folder, annotations_dir=training_annotations,
                                batch_size=batch_size,
                                use_vgg=False, shuffle=True,
                                load_annotations=False,
                                img_size=img_size,
                                load_color_imgs=False,
                                num_of_instance=10,
                                use_fusion=False)

valid_generator = DataGenerator(valid_images_folder,
                                annotations_dir=testing_annotations,
                                batch_size=batch_size,
                                use_vgg=False,
                                shuffle=True,
                                img_size=img_size,
                                load_annotations=False,
                                load_color_imgs=False,
                                num_of_instance=10,
                                use_fusion=False)

# %% Load Model

full_colorization_model = Full_ImageColorization(input_shape)
# load predicts optional
recent_weights = file_paths.recent_weights
full_colorization_model.load_weights(recent_weights)


# %% define smooth l1 loss function
def smooth_L1_loss(y_true, y_pred):
    return tf.compat.v1.losses.huber_loss(y_true, y_pred)


# %% complie model
optimizer = keras.optimizers.RMSprop(lr=5e-5)
full_colorization_model.compile(optimizer=optimizer, loss=smooth_L1_loss)

checkpoint_path = file_paths.checkpoint_path
# Keras callbacks for training
callback_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path + \
                 "FullImage_weights.e{epoch:02d}-" + \
                 "loss{val_loss:.4f}.h5",
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True),
    # tf.keras.callbacks.LearningRateScheduler(scheduler),
]
# %% train full model
model_history = full_colorization_model.fit(train_generator, epochs=epochs,
                                            validation_data=valid_generator,
                                            callbacks=callback_list, shuffle=True)

# %% Predictions
# load predicts optional
recent_weights = file_paths.recent_weights
full_colorization_model.load_weights(recent_weights)

# %% generate data
stop = np.random.randint(0, batch_size)

for i, (grayimgs, color_imgs) in enumerate(valid_generator):
    gray_images = grayimgs
    target_images = color_imgs
    if i == stop:
        break

# %% prediction
predictions = full_colorization_model.predict(gray_images[0])
# convert to rgb
rgb_predict_color_imgs, rgb_target_color_imgs = rgb_color_img_batches(gray_images, predictions, target_images,
                                                                      batch_size - 15)


# %% show image
def show_imgs(num):
    fig = plt.figure()
    ax = fig.subplots(1, 2)
    ax[0].imshow(rgb_target_color_imgs[num, :, :, :])
    ax[0].set_title("Real_image")
    ax[1].imshow(rgb_predict_color_imgs[num, :, :, :])
    ax[1].set_title("predicted_image")
    plt.axis("off")


# %% display image

num = 1
show_imgs(num)
