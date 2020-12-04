# This file pre-processes our data to feed into our network
# There generator gets data for each of our networks: full image, instance and fusion
# The pre-processing includes: shuffling dataset, batch size, number of instances per image, option to use pre-trained network
# Images were converted to lab space such that the intensity of the pixels are decoupled from color

#%% Import libraries
import os
from skimage import io,color,img_as_float32
from skimage.transform import resize
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.applications.vgg16 import preprocess_input
from helper import crop_images

#%% Custom Class
# Attempts to use pre-trained weights like 'use_vgg' led to poor results and were dropped after a few experimentations
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, annotations_dir=None, batch_size=32, num_of_instance=3, shuffle=True, img_size=224, use_vgg=False, 
                 load_annotations=False, load_color_imgs=False, use_fusion = False):
        # Image directory
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        # Annotation directory
        self.annotations_dir = annotations_dir
        self.annotations_files = np.sort(os.listdir(annotations_dir))
        # Maximum number of images we wish to retrieve from one image, else the miniumum number will be retrieved
        self.num_of_instance = num_of_instance
      
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = img_size
        self.load_color_imgs = load_color_imgs
        self.load_annotations = load_annotations
        self.use_vgg = use_vgg
        self.use_fusion = use_fusion
        self.on_epoch_end()
        

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    # Function to get data for the network chosen: full, instance, fusion
    # If parameter load_annotations is set to true, images for the instance network will be retrieved
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch = [self.image_files[k] for k in indexes]
        
        if self.load_annotations:
            annotations = [self.annotations_files[k] for k in indexes]
            gray_scale, colors_channels = self.__get_annotate_data(batch,annotations)
            return gray_scale, colors_channels
            
        elif self.use_fusion:
            annotations = [self.annotations_files[k] for k in indexes]
            inputs,outputs = self.__get_fusion_data(batch,annotations)
            
            return inputs,outputs
            
        else:
            gray_scale, colors_channels = self.__get_data(batch) 
            return gray_scale, colors_channels

    # Shuffle dataset indicies after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    
    # This function gets data for the full image colorization network
    def __get_data(self, batch):
        
        color_channel_imgs = np.zeros((self.batch_size, self.img_size,self.img_size,2))
        color_imgs =  np.zeros((self.batch_size, self.img_size,self.img_size,3))
        
        # Using pre-trained vgg was scrapped after a few experiments but we leave it here in case 
        # any one else would like to experiment with it
        if self.use_vgg:
            gray_scale_imgs = np.zeros((self.batch_size, self.img_size,self.img_size,3))
        else:
            gray_scale_imgs = np.zeros((self.batch_size, self.img_size,self.img_size,1))
            
        # Resizing images in our batch to that of the size we passed into our network
        # Image size of 128x128 was used in our final implementation
        for i, id in enumerate(batch):
            image_name = os.path.join(self.image_dir, id)  
            imagergb = io.imread(image_name)
            imagergb = resize(imagergb,(self.img_size,self.img_size,3),mode="reflect")  #resize image
            
             
            if imagergb.ndim==2:
                continue
                r = np.zeros((imagergb.shape[0],imagergb.shape[1],3))
                r[:,:,0] = imagergb
                r[:,:,1] = imagergb
                r[:,:,2] = imagergb
                imagergb = r
            color_img = imagergb

            # Convert to lab space
            lab = img_as_float32(color.rgb2lab(imagergb)) 
            
            
            # Because VGG accepts 3 channels, 1 channel gray images are converted to 3 as below
            if self.use_vgg: 
                gray_scale = np.zeros_like(lab)
                grayx = (lab[:,:,0]/100).astype("float64") #range 0 to 1
                gray_scale[:,:,0] = grayx
                gray_scale[:,:,1] = grayx
                gray_scale[:,:,2] = grayx
            else:
                gray_scale = (lab[:,:,0]/100).astype("float64")[:,:,np.newaxis] #range 0 to 1
                
            color_channels = ((lab[:,:,1:]/128)).astype("float64") #range -1 to 1
            
            gray_scale_imgs[i,:] = gray_scale #for vgg model
            color_channel_imgs[i,:] = color_channels
            color_imgs[i,:] = color_img.astype("float64")
            
        if self.load_color_imgs:
            return gray_scale_imgs,[color_channel_imgs,color_imgs]
        else:
            return gray_scale_imgs,color_channel_imgs
        
    # This function gets data for the instance image colorization network
    def __get_annotate_data(self, batch, annotations):
        # Arrays to hold varying amounts of instance images from the batch images
        # For example in a batch of 2 images, image 1 may have 3 instances and image 2 may have 5 instances
        gray_imgs =[]
        color_imgs = []

        for i, (id,id2) in enumerate(zip(batch,annotations)):
            
            image_name = os.path.join(self.image_dir, id)  
            imagergb = io.imread(image_name)
            # Image scores were loaded as well, this was to reduce simple textures from being passed through
            # For example image textures like patches of objects like bread, grass, sky typically had a score lower than 
            # objects like people, kitchenware etc. 
            boxx = np.load(os.path.join(self.annotations_dir, id2))['bbox'] 
            scores = np.load(os.path.join(self.annotations_dir, id2))['scores']
            crop_imgs,_ = crop_images(imagergb, boxx,scores, self.num_of_instance) #number of instances to use
            gray_img = []
            color_img =[]
            
            for j,images in enumerate(crop_imgs):
                # Convert each instance detected in the full image into the resized shape (128x128 used in final implementation)
                images = resize(images,(self.img_size,self.img_size,3),mode="reflect")
                if images.ndim==2:
                    r = np.zeros((images.shape[0],images.shape[1],3))
                    r[:,:,0] = images
                    r[:,:,1] = images
                    r[:,:,2] = images
                    images = r
                # Convert each instance detected in the full image into lab space 
                lab = img_as_float32(color.rgb2lab(images))
                
                if self.use_vgg: #since vgg accept 3 channels
                    gray_scale = np.zeros_like(lab)
                    grayx = (lab[:,:,0]/100).astype("float64") #range 0 to 1
                    gray_scale[:,:,0] = grayx
                    gray_scale[:,:,1] = grayx
                    gray_scale[:,:,2] = grayx
                else:
                    gray_scale = (lab[:,:,0]/100).astype("float64")[:,:,np.newaxis] #range 0 to 1
                   
                color_channels = (lab[:,:,1:]/128).astype("float64") #range -1 to 1
                
                gray_img.append(gray_scale)
                color_img.append(color_channels)
            

            if len(gray_img) < 1:
                continue
  
            gray_imgs.append(np.asarray(gray_img))
            color_imgs.append(np.asarray(color_img))
        
      
        return gray_imgs, color_imgs
    
    # This function gets data for the fusion image colorization network
    def __get_fusion_data(self, batch, annotations):
        # For the fusion network, a batch size of only 1 is used
        # Batch sizes greater than 1 led to an unsuccessful complex implementations of matching instance with its full image
        # The original paper also used a batch size of 1 in their implementation
        color_channel_imgs = np.zeros((self.batch_size, self.img_size,self.img_size,2))
        
        if self.use_vgg:
            gray_scale_imgs = np.zeros((self.batch_size, self.img_size,self.img_size,3))
        else:
            gray_scale_imgs = np.zeros((self.batch_size, self.img_size,self.img_size,1))
            
        # Instance images variables to keep track for fusion
        igray_imgs = []
        icolor_imgs = []
        ibounding_boxes = []
        iimage_initial_sizes = []
        
        for i, (id,id2) in enumerate(zip(batch,annotations)):
            image_name = os.path.join(self.image_dir, id)  
            imagergb = io.imread(image_name)
            im_real = imagergb.copy()
            
            # Image resizing
            fullimagergb = resize(imagergb,(self.img_size,self.img_size,3),mode="reflect") 
            if fullimagergb.ndim==2:
                r = np.zeros((fullimagergb.shape[0],fullimagergb.shape[1],3))
                r[:,:,0] = fullimagergb
                r[:,:,1] = fullimagergb
                r[:,:,2] = fullimagergb
                fullimagergb = r
            
            # Convert full image to lab space
            lab = img_as_float32(color.rgb2lab(fullimagergb)) 
            
            if self.use_vgg: #since vgg accept 3 channels
                gray_scale = np.zeros_like(lab)
                grayx = (lab[:,:,0]/100).astype("float64") #range 0 to 1
                gray_scale[:,:,0] = grayx
                gray_scale[:,:,1] = grayx
                gray_scale[:,:,2] = grayx
            else:
                gray_scale = (lab[:,:,0]/100).astype("float64")[:,:,np.newaxis] #range 0 to 1
                    
            color_channels = (lab[:,:,1:]/128).astype("float64") #range -1 to 1
            
            gray_scale_imgs[i,:] = gray_scale #for vgg model
            color_channel_imgs[i,:] = color_channels
            
            # Scores used for the same reason as the instance image network, to prevent training on textures / ambiguous objects
            boxx = np.load(os.path.join(self.annotations_dir, id2))['bbox'] 
            scores = np.load(os.path.join(self.annotations_dir, id2))['scores']
            crop_imgs,bound_box  = crop_images(imagergb, boxx,scores, self.num_of_instance)  #number of instances to use
            
            igray_img = []
            icolor_img =  []
            initial_im_size = []
            
            # For loop for operations on instance images 
            for j,images in enumerate(crop_imgs):
                
                images = resize(images,(self.img_size,self.img_size,3),mode="reflect")
                if images.ndim==2:
                    ir = np.zeros((images.shape[0],images.shape[1],3))
                    ir[:,:,0] = images
                    ir[:,:,1] = images
                    ir[:,:,2] = images
                    images = ir

                # convert each instance image to lab space
                lab2 = img_as_float32(color.rgb2lab(images))
                
                if self.use_vgg: #since vgg accept 3 channels
                    igray_scale = np.zeros_like(lab2)
                    igrayx = (lab[:,:,0]/100).astype("float64") #range 0 to 1
                    igray_scale[:,:,0] = igrayx
                    igray_scale[:,:,1] = igrayx
                    igray_scale[:,:,2] = igrayx
                else:
                    igray_scale = (lab2[:,:,0]/100).astype("float64")[:,:,np.newaxis] #range 0 to 1
                   
                
                icolor_channels = (lab2[:,:,1:]/128).astype("float64") #range -1 to 1
                
                igray_img.append(igray_scale)
                icolor_img.append(icolor_channels)
                initial_im_size.append(im_real.shape)
                
            if len(igray_img) < 1:
                igray_img=np.zeros((1,self.img_size,self.img_size,1), )
                icolor_img = np.zeros((1,self.img_size,self.img_size,2))
                bound_box = np.zeros((1,4))
                bound_box[0,:] =np.array([10,10,20,20])
                initial_im_size.append((self.img_size,self.img_size))
            
            ibounding_boxes.append(bound_box)
            igray_imgs.append(np.asarray(igray_img))
            icolor_imgs.append(np.asarray(icolor_img))

            iimage_initial_sizes.append(initial_im_size)
            iimage_initial_sizes = np.squeeze(np.asarray(iimage_initial_sizes), axis=0)

            igray_imgs = np.squeeze(np.asarray(igray_imgs),axis=0)
            ibounding_boxes = np.squeeze(np.asarray(ibounding_boxes),axis=0).astype("float64")

        return (gray_scale_imgs,igray_imgs, ibounding_boxes, iimage_initial_sizes),color_channel_imgs