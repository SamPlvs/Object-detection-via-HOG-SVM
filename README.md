# Object-detection-via-HOG-SVM
This is an application of Object detection using Histogram of Oriented Gradients (HOG) as features and Support Vector Machines (SVM) 
as the classifier. 

This process is implemented in python, the following libraries are required:
1. Scikit-learn (For implementing SVM)
2. Scikit-image (For HOG feature extraction)
3. OpenCV (for testing)
4. PIL (Image processing library)
5. Numpy (matrix multiplication)
6. __ for Non-maximum suppression

A training set should comprise of:
1. Positive images: these images should contain only the object you are trying to detect
2. Negative images: these images can contain anything except for the object you are detecting

I have provided a link for the Inria dataset - for human detection but this code can be adapted for other datasets too i.e. car detection.

The files are divided into the following:
Training & Testing (this is where you evaluate your trained classifier)
