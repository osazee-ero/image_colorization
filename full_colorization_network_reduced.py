# This file describes the network that we used for our full colorization and instance colorization
# Both have the same architecture.
# Various architectures and experiments trialed are highlighted in the write up accompanying the code

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten, Lambda, Reshape, Concatenate, Softmax, Dense,BatchNormalization, ReLU, MaxPool2D,Input,Conv2DTranspose,UpSampling2D,Dropout,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np

#%% Define model

def Full_ImageColorization(img_size):

        input_shape = Input(img_size) #0
        
        convf = Conv2D(64, (4, 4), 1, padding='same')(input_shape) #128 x 128 x 64 #1
        
        convf = BatchNormalization()(convf) #2
        convf = LeakyReLU(0.2)(convf) #3
        
        conv1 = Conv2D(64, (4, 4), 2, padding='same')(convf) #64 x 64 x 64 #4
        conv1 = BatchNormalization()(conv1) #5
        conv1 = LeakyReLU(0.2)(conv1) #6

        conv2 = Conv2D(128, (4, 4), 2, padding='same')(conv1) #32 x 32 x 128 #7
        conv2 = BatchNormalization()(conv2) #8
        conv2 = LeakyReLU(0.2)(conv2) #9
        
        conv3 = Conv2D(256, (4, 4), 2, padding='same')(conv2) #16 x 16 x 256 #10
        conv3 = BatchNormalization()(conv3) #11
        conv3 = LeakyReLU(0.2)(conv3) #12
        
        conv4 = Conv2D(512, (4, 4), 2, padding='same')(conv3) #8 x 8 x 512 #13
        conv4 = BatchNormalization()(conv4) #14
        conv4 = LeakyReLU(0.2)(conv4) #15
        
        conv5 = Conv2D(512, (4, 4), 2, padding='same')(conv4) #4 x 4 x 512 #16
        conv5 = BatchNormalization()(conv5) #17
        conv5 = LeakyReLU(0.2)(conv5) #18
        
        conv6 = Conv2D(512, (4, 4), 2, padding='same')(conv5) #2 x 2 x 512 #19
        conv6 = BatchNormalization()(conv6) #20
        conv6 = LeakyReLU(0.2)(conv6) #21

        conv8 = Conv2DTranspose(512, (4, 4), 2, padding='same')(conv6) #4 x 4 x 512 #25
        conv8 = BatchNormalization()(conv8) #26
        conv8 = ReLU()(conv8) #27

        conv9 = Conv2DTranspose(512, (4, 4), 2, padding='same')(conv8) #8 x 8 x 512 #29
        conv9 = BatchNormalization()(conv9) #30
        conv9 = ReLU()(conv9) #31
     
        conv10 = Conv2DTranspose(512, (4, 4), 2, padding='same')(conv9) #16 x 16 x 512 #33
        conv10 = BatchNormalization()(conv10) #34
        conv10 = ReLU()(conv10) #35
 
        conv11 = Conv2DTranspose(256, (4, 4), 2, padding='same')(conv10) #32 x 32 x 256 #37
        conv11 = BatchNormalization()(conv11) #38
        conv11 = ReLU()(conv11) #39
 
        conv12 = Conv2DTranspose(128, (4, 4), 2, padding='same')(conv11) #64 x 64 x 128 #41
        conv12 = BatchNormalization()(conv12) #42
        conv12 = ReLU()(conv12) #43

        conv13 = Conv2DTranspose(64, (4, 4), 2, padding='same')(conv12) #128 x 128 x 64 #45
        conv13 = BatchNormalization()(conv13) #46
        conv13 = ReLU()(conv13) #47

        outputs_layer = Conv2D(2, (4, 4), 1, activation="tanh", padding='same')(conv13) #128 x 128 x 2 #53
        
        model = Model(input_shape, outputs_layer) 
                
        model.summary()
        print('fullimg')
        return model
    
    
  
#%% Instance recongition    


    
def Instance_ImageColorization(img_size):

        input_shape = Input(img_size) #0
        
        convf = Conv2D(64, (4, 4), 1, padding='same')(input_shape) #128 x 128 x 64 #1
        
        convf = BatchNormalization()(convf) #2
        convf = LeakyReLU(0.2)(convf) #3
        
        conv1 = Conv2D(64, (4, 4), 2, padding='same')(convf) #64 x 64 x 64 #4
        conv1 = BatchNormalization()(conv1) #5
        conv1 = LeakyReLU(0.2)(conv1) #6
        
        conv2 = Conv2D(128, (4, 4), 2, padding='same')(conv1) #32 x 32 x 128 #7
        conv2 = BatchNormalization()(conv2) #8
        conv2 = LeakyReLU(0.2)(conv2) #9
        
        conv3 = Conv2D(256, (4, 4), 2, padding='same')(conv2) #16 x 16 x 256 #10
        conv3 = BatchNormalization()(conv3) #11
        conv3 = LeakyReLU(0.2)(conv3) #12
        
        conv4 = Conv2D(512, (4, 4), 2, padding='same')(conv3) #8 x 8 x 512 #13
        conv4 = BatchNormalization()(conv4) #14
        conv4 = LeakyReLU(0.2)(conv4) #15
        
        conv5 = Conv2D(512, (4, 4), 2, padding='same')(conv4) #4 x 4 x 512 #16
        conv5 = BatchNormalization()(conv5) #17
        conv5 = LeakyReLU(0.2)(conv5) #18
        
        conv6 = Conv2D(512, (4, 4), 2, padding='same')(conv5) #2 x 2 x 512 #19
        conv6 = BatchNormalization()(conv6) #20
        conv6 = LeakyReLU(0.2)(conv6) #21
        
        conv8 = Conv2DTranspose(512, (4, 4), 2, padding='same')(conv6) #4 x 4 x 512 #25
        conv8 = BatchNormalization()(conv8) #26
        conv8 = ReLU()(conv8) #27
  
        conv9 = Conv2DTranspose(512, (4, 4), 2, padding='same')(conv8) #8 x 8 x 512 #29
        conv9 = BatchNormalization()(conv9) #30
        conv9 = ReLU()(conv9) #31
      
        conv10 = Conv2DTranspose(512, (4, 4), 2, padding='same')(conv9) #16 x 16 x 512 #33
        conv10 = BatchNormalization()(conv10) #34
        conv10 = ReLU()(conv10) #35

        conv11 = Conv2DTranspose(256, (4, 4), 2, padding='same')(conv10) #32 x 32 x 256 #37
        conv11 = BatchNormalization()(conv11) #38
        conv11 = ReLU()(conv11) #39
     
        conv12 = Conv2DTranspose(128, (4, 4), 2, padding='same')(conv11) #64 x 64 x 128 #41
        conv12 = BatchNormalization()(conv12) #42
        conv12 = ReLU()(conv12) #43
  
        conv13 = Conv2DTranspose(64, (4, 4), 2, padding='same')(conv12) #128 x 128 x 64 #45
        conv13 = BatchNormalization()(conv13) #46
        conv13 = ReLU()(conv13) #47
        
        outputs_layer = Conv2D(2, (4, 4), 1, activation="tanh", padding='same')(conv13) #128 x 128 x 2 #53
        
        model = Model(input_shape, outputs_layer) 
        model.summary()
            
        return model
