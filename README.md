# Object-detection-via-HOG-SVM
This is an application of Object detection using Histogram of Oriented Gradients (HOG) as features and Support Vector Machines (SVM) 
as the classifier. 

This process is implemented in python, the following libraries are required:
1. Scikit-learn (For implementing SVM)
2. Scikit-image (For HOG feature extraction)
3. OpenCV (for testing)
4. PIL (Image processing library)
5. Numpy (matrix multiplication)
6. Imutils for Non-maximum suppression

A training set should comprise of:
1. Positive images: these images should contain only the object you are trying to detect
2. Negative images: these images can contain anything except for the object you are detecting

Web link for the Inria dataset is shown below - the dataset is for pedestrian detection but this code can be adapted for other datasets too eg. car detection (dataset link: http://cogcomp.org/Data/Car/). Inria dataset link: http://pascal.inrialpes.fr/data/human/

The files are divided into the following:
Training & Testing (this is where you evaluate your trained classifier)
Visualise Hog: simply allows you to see what the gradients calculated look like on a given image (specified by the user).

The results of the trained person detector on a test image are as follows: Normal (RGB image)> HOG descriptors
![test](https://user-images.githubusercontent.com/35964759/38281042-9523760a-37a0-11e8-914d-917308e3ac22.png)

After classifying with a trained SVM model and applying NMS the following result is achieved:
![raw detections after nms](https://user-images.githubusercontent.com/35964759/38281024-75dcb50e-37a0-11e8-81fe-6df2dede1f78.png)

Note The results are better when the background is not cluttered (shown below). As seen from the original and the extracted HOG image, majority of the gradients binned come from the background (this could be due to the training set not being robust to more greener/cluttered background as shown in the test)

![image](https://user-images.githubusercontent.com/35964759/38281107-e92f8d60-37a0-11e8-951b-d1d316460386.png)

This method is also robust to different poses and presence of multiple subjects as shown in the figure below where an image from google was shown to the webcam
![svm_robust](https://user-images.githubusercontent.com/35964759/39673857-9c4452a6-513b-11e8-8e64-42b55e23c200.png)

