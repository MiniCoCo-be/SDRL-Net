# Scale-wise Discriminative Region Learning for Medical Image Segmentation


Access to the multi-organ dataset:
    Download the dataset from the [link](https://drive.google.com/drive/u/1/folders/1lV5dxLnthVCSLNIOWz63MqrZpGUoNCU8). 
    Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keep the 3D volume in h5 format for testing cases.
