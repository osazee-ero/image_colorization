## Prerequisites
1. Install [conda](https://www.anaconda.com/).
2. Create an environment
```sh
conda create --name virtual python=3.7
conda activate virtual
```
3. Install dependencies
* A NVIDIA® GPU card is needed check [CUDA® Enabled GPUs](https://developer.nvidia.com/cuda-gpus)
* For RTX 30 series cards, check [Build tensorflow from source](https://www.tensorflow.org/install/source) to install tensorflow for CUDA® 11.1
```sh
pip install numpy pandas matplotlib scikit-image
pip install opencv-python==3.4.11.41 opencv-contrib-python==3.4.11.41
conda install -c anaconda cudatoolkit==10.1 cudnn
pip install tensorflow-gpu scikit-learn
```

## Getting Started
1. Clone this repo:
```sh
git clone https://github.com/r444li/SYDE671-final_project-image_colorization.git
```
2. Download [dataset](http://images.cocodataset.org/zips/train2017.zip) (~18GB).
3. Download [files of bounding boxes](https://drive.google.com/file/d/19LbOdkt9dy7RyFp4kNpyu0-mLzFhECYU/view?usp=sharing).
4. Create **4** directories for training images and annotations, validating images and annotations
5. Create **1** directory for checkpoints
6. Update paths in **file_paths.py**
7. Put all the images downloaded in the directory of **training images**
8. Put all the files of bounding boxes downloaded in the direcotry of **train annotations**
7. Run **clean_images.py**


## Training Full Model
1. Set **batch_size**, **epochs**, **img_size** in **Train_Full_ImageColorizationNetwork_reduced.py**(Line 17-19)
2. Run **Train_Full_ImageColorizationNetwork_reduced.py**

## Training Instance Model
1. Set **batch_size**, **epochs**, **img_size** in **Train_Instance_ImageColorizationNetwork_reduced.py**(Line 17-19)
2. Run **Train_Instance_ImageColorizationNetwork_reduced.py**

## Training Fusion Model
* Before train fusion model, there must be at least one file of weights for **each of full and instance models** in the directory of **checkpoints**
1. Choose one weights file for **each of full and instance models** in the directory of **checkpoints**.
2. Update **full_imagecolorization_weights** and **instance_imagecolorization_weights** in **file_paths.py**
3. Set **epochs**, **img_size** in **fusion_train_reduced.py**(Line 22-23)
4. Run **fusion_train_reduced.py**

## View the results of models
1. Choose one weights file in the directory of **checkpoints**.
2. Create **1** directory for outputs
3. Update **recent_weights** and **output_path** in **file_paths.py** to the chosed file.
4. Update **train_generator** and **valid_generator** in **test.py**(start at Line 30) to the definition in the training program of chosed model.
5. Run **test.py**

