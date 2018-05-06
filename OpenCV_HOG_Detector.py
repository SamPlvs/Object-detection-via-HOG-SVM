# Modified from OpenCV HOG person detector (should work straight off the bat)

import numpy as np
import cv2
import sys
from glob import glob
import itertools as it

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 0, 255), thickness)

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
''' the above code uses a pretrained SVM via HOG descriptors provided by the open cv database.
This database is limited to the training it has performed hence cannot be used in any other angle other than perp. to the centroid
Thus if you want to implement the HOG + SVM method, you'll have to train your own SVM with your own data'''
cap= cv2.VideoCapture(0)
# the above code uses the OpenCV library to capture video frames from the camera: select 0 for the primary pc webcam & 1 for an external camera

while True:
    #running an infinite loop so that the process is run real time.
    ret, img = cap.read() # reading the frames produced from the webcam in 'img' an then returning them using the 'ret' function.
    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05) # describing the parameters of HOG and returning them as a Human found function in 'found'
    found_filtered = [] #filtering the found human... to further improve visualisation (uses Gaussian filter for eradication of errors produced by luminescence.
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
            else:
                found_filtered.append(r)
        draw_detections(img, found) # using the predefined bounding box to encapsulate the human detected within the bounding box.
        draw_detections(img, found_filtered, 3) # further filtering the box to improve visualisation.
        print('%d (%d) found' % (len(found_filtered), len(found))) # this will produce the output of the number of humans found in the actual command box)
    cv2.imshow('img', img) # finally showing the resulting image captured from the webcam.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # defining a key to quit and stop all processes. The key is 'q'
cap.release()
cv2.destroyAllWindows() # finally, destroying all open windows.
