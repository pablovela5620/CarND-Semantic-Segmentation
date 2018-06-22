# Semantic Segmentation
### Introduction
In this project, I built a Fully Convolutional Network to label the pixels of a road in images.

![Simulation Gif](https://media.giphy.com/media/w6yTuQ4Yuh2aRURlFs/giphy.gif)

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - [OpenCV](https://opencv.org/)
 
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images. I renamed `data_road` to `images` and also the groundtruth labels to match the names of the images. This was to simply the code for the image/label generator.
