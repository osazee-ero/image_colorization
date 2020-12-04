# Use this script to train the fusion network
# This is meant to be done after training the full image and instance network

#%% Imports

from full_colorization_network_reduced import Full_ImageColorization, Instance_ImageColorization
from FusionModel_reduced import FusionNetwork
from CustomDataGenerator import  DataGenerator
from helper import rgb_color_img_batches
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import file_paths

#%% Parameters

bound_box = (4,)
origin_img_size = (3,)
input_shape = (128,128,1,)
batch_size = 1
epochs = 2
img_size = 128


#%% Load Data

images_folder = file_paths.images_folder  #training set
valid_images_folder = file_paths.valid_images_folder  #validation set

training_annotations = file_paths.training_annotations
testing_annotations = file_paths.testing_annotations


# Data generators
train_generator = DataGenerator(images_folder,annotations_dir=training_annotations, 
                                batch_size=batch_size,
                                use_vgg=False, shuffle=True, 
                                load_annotations=False, 
                                img_size=img_size, 
                                load_color_imgs=False,
                                num_of_instance=3,
                                use_fusion=True)

valid_generator  = DataGenerator(valid_images_folder,
                                 annotations_dir=testing_annotations,
                                 batch_size=batch_size, 
                                 use_vgg=False, 
                                 shuffle=True, 
                                 img_size=img_size, 
                                 load_annotations=False, 
                                 load_color_imgs=False,
                                 num_of_instance=3,
                                 use_fusion=True)

#%% load weights for both model

full_imagecolorization_weights = file_paths.full_imagecolorization_weights
instance_imagecolorization_weights = file_paths.instance_imagecolorization_weights

full_imageColorizationModel = Full_ImageColorization(input_shape)
instance_imageColorizationModel = Instance_ImageColorization(input_shape)

full_imageColorizationModel.load_weights(full_imagecolorization_weights)
instance_imageColorizationModel.load_weights(instance_imagecolorization_weights)

#%% Set weights to untrainable in the full image and instance networks
for layers in full_imageColorizationModel.layers:
    layers.trainable = False
    
for layers in instance_imageColorizationModel.layers:
    layers.trainable = False
    
#%% check model summary
full_imageColorizationModel.summary()
instance_imageColorizationModel.summary()

#%% Fusion Model
fusion_model = FusionNetwork(full_imageColorizationModel,instance_imageColorizationModel, origin_img_size, bound_box)
fusion_model.load_weights("/home/andy/Documents/InstColorization/fromScratch/full_colorization_model_check_points/fusionwInstAdded.h5")


#%% define smooth l1 loss function 
def smooth_L1_loss(y_true, y_pred):
    return tf.compat.v1.losses.huber_loss(y_true, y_pred)

#%% Compile model
optimizer = keras.optimizers.RMSprop(lr=5e-5)
fusion_model.compile(optimizer=optimizer, loss=smooth_L1_loss)

checkpoint_path = file_paths.checkpoint_path
# Keras callbacks for training
callback_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path + \
                "Fusion_weights.e{epoch:02d}-" + \
                "loss{val_loss:.4f}.h5",
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True)
            ]

tf.keras.backend.set_floatx('float32')

#%% train full model
model_history = fusion_model.fit(train_generator, epochs=epochs, 
                                                      validation_data=valid_generator,
                                                      shuffle=True,
                                                       )
# The model will not save without the line below
# This is only true for training this network and not for others
fusion_model.save('/home/andy/Documents/InstColorization/fromScratch/full_colorization_model_check_points/fusionwInstAdded.h5')


#%% generate data
stop = np.random.randint(0,batch_size)

#%% test fusion generator here

for i,((gray_scale_imgs,igray_imgs, x, y),color_channel_imgs) in enumerate(train_generator):
    full_image_gray,instance_image_gray,full_image_targets, x, y = gray_scale_imgs,igray_imgs,color_channel_imgs, x, y
    if i==stop:
        break

#%% prediction
print(full_image_gray.shape, instance_image_gray.shape)
predictions = fusion_model.predict_on_batch([full_image_gray,instance_image_gray, x, y])

full_image_gray = np.tile(full_image_gray, (instance_image_gray.shape[0], 1,1,1))
full_image_targets = np.tile(full_image_targets, (full_image_targets.shape[0], 1,1,1))

# #convert to rgb
print(predictions.shape)
print(full_image_gray.shape, full_image_targets.shape)

rgb_predict_color_imgs, rgb_target_color_imgs =  rgb_color_img_batches(full_image_gray, predictions, full_image_targets, 1)

#%% show image
def show_imgs(num):
    fig = plt.figure()
    ax = fig.subplots(1,2)
    ax[0].imshow(rgb_target_color_imgs[num,:,:,:])
    ax[0].set_title("Real_image")
    ax[1].imshow(rgb_predict_color_imgs[num,:,:,:])
    ax[1].set_title("predicted_image")
    plt.axis("off")
#%% display image   
num=0
show_imgs(num)
