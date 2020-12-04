# This file describes the network that we used for our fusion module
# Various architectures and experiments trialed are highlighted in the write up accompanying the code
# Below also includes commented code for our fusion attempt
# We were seeing odd outputs as highlighted in the report and thus we resorted to a simple weighted fusion of the networks

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Lambda, Reshape, Concatenate, Softmax, Dense,BatchNormalization, ReLU, MaxPool2D,Input,Conv2DTranspose,UpSampling2D,Dropout,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import numpy as np
from skimage.transform import resize

#%% simple model

class SimpleWeightModel(layers.Layer):
    def __init__(self,num_filters):
        super(SimpleWeightModel, self).__init__()
        
        self.simple_instance_conv =   Sequential(
                                                [Conv2D(num_filters, 4, 1, padding="same"),
                                                BatchNormalization(),
                                                LeakyReLU(0.2),
                                                Conv2D(num_filters, 4, 1, padding="same"),
                                                BatchNormalization(),
                                                LeakyReLU(0.2),
                                                Conv2D(1, 4, 1, padding="same"),
                                                BatchNormalization(),
                                                LeakyReLU(0.2)]
                                                 )
        self.simple_bg_conv =         Sequential(
                                                [Conv2D(num_filters, 4, 1, padding="same"),
                                                BatchNormalization(),
                                                LeakyReLU(0.2),
                                                Conv2D(num_filters, 4, 1, padding="same"),
                                                BatchNormalization(),
                                                LeakyReLU(0.2),
                                                Conv2D(1, 4, 1, padding="same"),
                                                BatchNormalization(),
                                                LeakyReLU(0.2)]
                                                 )
        self.normalize = Softmax(axis=-1)

    # This needed to be added to prevent errors in training and saving the model
    # StackOverflow reference: 
    # https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'simple_instance_conv': self.simple_instance_conv,
            'simple_bg_conv': self.simple_bg_conv,
            'normalize': self.normalize
        })
        return config
        
       
    # def resize_and_zero_pad(self,weight_maps,bound_box, target_img_width, target_img_height, original_img_sizes):
    #     # weight_maps, bound_box, original_img_sizes, target_img_width, target_img_height = inputs
    #     num=tf.shape(weight_maps)[0]
    #     # weight_maps_resized = tf.zeros_like(weight_maps)
    #     weight_maps_resized = []
    #     for i in range( num):
    #         x1,y1,x2,y2 = bound_box[i,:]
    #         im_height, im_width = original_img_sizes[i,0],original_img_sizes[i,1]
    #         resize_wmaps = tf.image.resize(weight_maps[i,:],[y2-y1, x2-x1]) #resize weight maps
    #         # print('wmap:', resize_wmaps.shape)
    #         # resize_wmaps = tf.pad(resize_wmaps, )
    #         # print(resize_wmaps.shape)
    #         # img_local_pos = tf.zeros((im_height, im_width,tf.shape(weight_maps)[-1]))

    #         ####### changed tf.variable to np.array and added dtype as float 32 ##########
    #         # img_local_pos = np.array((img_local_pos),dtype= np.float32)
    #         print('initial:', img_local_pos.shape, img_local_pos.dtype)

    #         # print('img shape', img_local_pos.shape)

    #         ####### removed .assign from end of [args] ##########
    #         # img_local_pos = np.array(img_local_pos)[:,:,np.newaxis]
    #         # img_local_pos[int(y1):int(y2),int(x1):int(x2), :] = resize_wmaps
    #         # print('img pos', img_local_pos[int(y1):int(y2),int(x1):int(x2), :].shape)
    #         # img_local_pos[y1:y2, x1:x2, :] = resize_wmaps  #localized in maps
            
    #         img_local_pos = tf.Variable((img_local_pos),dtype=tf.float32)
    #         img_local_pos = tf.convert_to_tensor(img_local_pos, dtype=tf.float32)
    #         img_local_pos = tf.image.resize(weight_maps[i,:], (target_img_width, target_img_height))
    #         print('final:', img_local_pos.shape, img_local_pos.dtype)
    #         weight_maps_resized.append(img_local_pos)

    #         # print("I got here")
    #         # if i == 0:
    #         #     weight_maps_resized = img_local_pos
    #         # else:
       
    #     # weight_maps_resized = np.asarray(weight_maps_resized, dype="float32")
    #     # weight_maps_resized = tf.convert_to_tensor(weight_maps_resized , dtype=tf.float32)
    #     weight_maps_resized = tf.stack(weight_maps_resized)
        
    #     return weight_maps_resized
    
    # def tf_function(self,weight_maps,bound_box, target_img_width, target_img_height, original_img_sizes):
    #      weight_maps_resized = tf.py_function(self.resize_and_zero_pad, [weight_maps,bound_box,target_img_width, target_img_height, original_img_sizes], tf.float32)
    #      weight_maps_resized.set_shape(weight_maps[0].get_shape())
         
    #      return weight_maps_resized


    def call(self, inputs):
        full_feat_maps, instance_feat_maps, bound_box, original_img_sizes = inputs
        
        full_weight_maps = self.simple_bg_conv(full_feat_maps) #1
        # full_weight_maps = self.normalize(full_weight_maps)
        # full_weight_maps_width,full_weight_maps_height = full_weight_maps.shape[1],full_weight_maps.shape[2]
        
        instance_weight_maps = self.simple_instance_conv(instance_feat_maps) #2
        # instance_weight_maps_resized = self.tf_function(instance_weight_maps,bound_box, full_weight_maps_width,full_weight_maps_height, original_img_sizes)
        # instance_weight_maps = self.normalize(instance_weight_maps)
        
        # print(instance_weight_maps.shape, instance_feat_maps.shape, "\n", full_weight_maps.shape, full_feat_maps.shape)

        # instance_feat_maps = tf.math.reduce_sum(instance_feat_maps, axis=-1)
        # instance_feat_maps = tf.expand_dims(instance_feat_maps, axis=-1)
        
        # instance_feat_maps_resized = self.tf_function(instance_feat_maps, bound_box, full_weight_maps_width,full_weight_maps_height)
    
        #multiply instance weights
        instance_operation1 = instance_weight_maps * instance_feat_maps

        # instance_operation1 = tf.math.reduce_sum(instance_operation1, axis=0)
        # instance_operation1 = tf.expand_dims(instance_operation1, axis=0)
        
        #multiply full weights
        full_operation1 = full_weight_maps * full_feat_maps
        
        #add weights together
        alpha = 0.2
        fused_features = full_operation1 + instance_operation1
        
        return fused_features
    

#%% fusion model

def FusionNetwork(full_imageColorizationModel,instance_imageColorizationModel, origin_img_size, bound_box):

    original_image_size = Input(shape=(origin_img_size))
    
    bounding_box = Input(shape=(bound_box))
    
    full_input_shape = full_imageColorizationModel.layers[0].output #0
    # full_input_shape = tf.keras.backend.squeeze(full_input_shape, axis=0)
  
    instance_input_shape = instance_imageColorizationModel.layers[0].output #0
    # instance_input_shape = tf.keras.backend.squeeze(instance_input_shape, axis=0)

    convf = full_imageColorizationModel.layers[1](full_input_shape) #128 x 128 x 64 #1
    convf = full_imageColorizationModel.layers[2](convf) #2
    convf = full_imageColorizationModel.layers[3](convf) #3
 
    iconvf = instance_imageColorizationModel.layers[1](instance_input_shape) #128 x 128 x 64 #1
    iconvf = instance_imageColorizationModel.layers[2](iconvf) #2
    iconvf = instance_imageColorizationModel.layers[3](iconvf) #3
  
    fusion0 = SimpleWeightModel(64)
    fusedFeat0 = fusion0([convf, iconvf, bounding_box, original_image_size])

    conv1 = full_imageColorizationModel.layers[4](fusedFeat0) #64 x 64 x 64 #4 
    conv1 = full_imageColorizationModel.layers[5](conv1) #5
    conv1 = full_imageColorizationModel.layers[6](conv1) #6
   
    iconv1 = instance_imageColorizationModel.layers[4](iconvf) #64 x 64 x 64 #4
    iconv1 = instance_imageColorizationModel.layers[5](iconv1) #5
    iconv1 = instance_imageColorizationModel.layers[6](iconv1) #6
    
    #weight map fusion layer
    fusion1 = SimpleWeightModel(64)
    fusedFeat1 = fusion1([conv1, iconv1, bounding_box, original_image_size])
    
    conv2 = full_imageColorizationModel.layers[7](fusedFeat1) #32 x 32 x 128 #7 
    conv2 = full_imageColorizationModel.layers[8](conv2) #8
    conv2 = full_imageColorizationModel.layers[9](conv2) #9
 
    iconv2 = instance_imageColorizationModel.layers[7](iconv1) #32 x 32 x 128 #7
    iconv2 = instance_imageColorizationModel.layers[8](iconv2) #8
    iconv2 = instance_imageColorizationModel.layers[9](iconv2) #9
  
    #weight map fusion layer
    fusion2 = SimpleWeightModel(64)
    fusedFeat2 = fusion2([conv2, iconv2, bounding_box, original_image_size])
    
    conv3 = full_imageColorizationModel.layers[10](fusedFeat2) #16 x 16 x 256 #10 
    conv3 = full_imageColorizationModel.layers[11](conv3) #11
    conv3 = full_imageColorizationModel.layers[12](conv3) #12
 
    iconv3 = instance_imageColorizationModel.layers[10](iconv2) #16 x 16 x 256 #10
    iconv3 = instance_imageColorizationModel.layers[11](iconv3) #11
    iconv3 = instance_imageColorizationModel.layers[12](iconv3) #12
    
    #weight map fusion layer
    fusion3 = SimpleWeightModel(64)
    fusedFeat3 = fusion3([conv3, iconv3, bounding_box, original_image_size])
    
    conv4 = full_imageColorizationModel.layers[13](fusedFeat3) #8 x 8 x 512 #13 
    conv4 = full_imageColorizationModel.layers[14](conv4) #14
    conv4 = full_imageColorizationModel.layers[15](conv4) #15

    iconv4 = instance_imageColorizationModel.layers[13](iconv3) #8 x 8 x 512 #13
    iconv4 = instance_imageColorizationModel.layers[14](iconv4) #14
    iconv4 = instance_imageColorizationModel.layers[15](iconv4) #15

    #weight map fusion layer
    fusion4 = SimpleWeightModel(64)
    fusedFeat4 = fusion4([conv4, iconv4, bounding_box, original_image_size])
    
    conv5 = full_imageColorizationModel.layers[16](fusedFeat4) #4 x 4 x 512 #16 
    conv5 = full_imageColorizationModel.layers[17](conv5) #17
    conv5 = full_imageColorizationModel.layers[18](conv5) #18
 
    iconv5 = instance_imageColorizationModel.layers[16](iconv4) #4 x 4 x 512 #16
    iconv5 = instance_imageColorizationModel.layers[17](iconv5) #17
    iconv5 = instance_imageColorizationModel.layers[18](iconv5) #18
 
    #weight map fusion layer
    fusion5 = SimpleWeightModel(64)
    fusedFeat5 = fusion5([conv5, iconv5, bounding_box, original_image_size])
    
    conv6 = full_imageColorizationModel.layers[19](fusedFeat5) #2 x 2 x 512 #19
    conv6 = full_imageColorizationModel.layers[20](conv6) #20
    conv6 = full_imageColorizationModel.layers[21](conv6) #21

    iconv6 = instance_imageColorizationModel.layers[19](iconv5) #2 x 2 x 512 #19
    iconv6 = instance_imageColorizationModel.layers[20](iconv6) #20
    iconv6 = instance_imageColorizationModel.layers[21](iconv6) #21
  
    #weight map fusion layer
    fusion6 = SimpleWeightModel(64)
    fusedFeat6 = fusion6([conv6, iconv6, bounding_box, original_image_size])
 
    conv8 = full_imageColorizationModel.layers[22](fusedFeat6) #4 x 4 x 512 #25
    conv8 = full_imageColorizationModel.layers[23](conv8) #26
    conv8 = full_imageColorizationModel.layers[24](conv8) #27

    iconv8 = instance_imageColorizationModel.layers[22](iconv6) #4 x 4 x 512 #25
    iconv8 = instance_imageColorizationModel.layers[23](iconv8) #26
    iconv8 = instance_imageColorizationModel.layers[24](iconv8) #27

    #weight map fusion layer
    fusion8 = SimpleWeightModel(64)
    fusedFeat8 = fusion8([conv8, iconv8, bounding_box, original_image_size])

    conv9 = full_imageColorizationModel.layers[25](fusedFeat8) #8 x 8 x 512 #29
    conv9 = full_imageColorizationModel.layers[26](conv9) #30
    conv9 = full_imageColorizationModel.layers[27](conv9) #31

    iconv9 = instance_imageColorizationModel.layers[25](iconv8) #8 x 8 x 512 #29
    iconv9 = instance_imageColorizationModel.layers[26](iconv9) #30
    iconv9 = instance_imageColorizationModel.layers[27](iconv9) #31
   
    #weight map fusion layer
    fusion9 = SimpleWeightModel(64)
    fusedFeat9 = fusion9([conv9, iconv9, bounding_box, original_image_size])
   
    conv10 = full_imageColorizationModel.layers[28](fusedFeat9)  #16 x 16 x 512 #33
    conv10 = full_imageColorizationModel.layers[29](conv10) #34
    conv10 = full_imageColorizationModel.layers[30](conv10) #35
   
    iconv10 = instance_imageColorizationModel.layers[28](iconv9)  #16 x 16 x 512 #33
    iconv10 = instance_imageColorizationModel.layers[29](iconv10) #34
    iconv10 = instance_imageColorizationModel.layers[30](iconv10) #35
  
    #weight map fusion layer
    fusion10 = SimpleWeightModel(64)
    fusedFeat10 = fusion10([conv10, iconv10, bounding_box, original_image_size])
    
    conv11 = full_imageColorizationModel.layers[31](fusedFeat10)  #32 x 32 x 256 #37
    conv11 = full_imageColorizationModel.layers[32](conv11) #38
    conv11 = full_imageColorizationModel.layers[33](conv11) #39

    iconv11 = instance_imageColorizationModel.layers[31](iconv10)  #32 x 32 x 256 #37
    iconv11 = instance_imageColorizationModel.layers[32](iconv11) #38
    iconv11 = instance_imageColorizationModel.layers[33](iconv11) #39
  
    #weight map fusion layer
    fusion11 = SimpleWeightModel(64)
    fusedFeat11 = fusion11([conv11, iconv11, bounding_box, original_image_size])
  
    conv12 = full_imageColorizationModel.layers[34](fusedFeat11)   #64 x 64 x 128 #41
    conv12 = full_imageColorizationModel.layers[35](conv12) #42
    conv12 = full_imageColorizationModel.layers[36](conv12) #43

    iconv12 = instance_imageColorizationModel.layers[34](iconv11)   #64 x 64 x 128 #41
    iconv12 = instance_imageColorizationModel.layers[35](iconv12) #42
    iconv12 = instance_imageColorizationModel.layers[36](iconv12) #43

    #weight map fusion layer
    fusion12 = SimpleWeightModel(64)
    fusedFeat12 = fusion12([conv12, iconv12, bounding_box, original_image_size])
 
    conv13 = full_imageColorizationModel.layers[37](fusedFeat12)   #128 x 128 x 64 #45
    conv13 = full_imageColorizationModel.layers[38](conv13) #46
    conv13 = full_imageColorizationModel.layers[39](conv13) #47

    iconv13 = instance_imageColorizationModel.layers[37](iconv12)   #128 x 128 x 64 #45
    iconv13 = instance_imageColorizationModel.layers[38](iconv13) #46
    iconv13 = instance_imageColorizationModel.layers[39](iconv13) #47

    #weight map fusion layer
    fusion13 = SimpleWeightModel(64)
    fusedFeat13 = fusion13([conv13, iconv13, bounding_box, original_image_size])
   
    outputs_layer = Conv2D(2, (4, 4), 1, activation="tanh", padding='same')(fusedFeat13) #128 x 128 x 2 #53
    
    model = Model([full_imageColorizationModel.input,instance_imageColorizationModel.input, bounding_box, original_image_size], outputs_layer) 
            
    model.summary()
    return model
