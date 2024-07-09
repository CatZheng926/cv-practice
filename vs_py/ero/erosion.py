import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('erosion.png')
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#cv_show('erosion',img)
kernel = np.ones((3,3),np.uint8)
erosion_1 = cv2.erode(img,kernel,iterations=1)
erosion_2 = cv2.erode(img,kernel,iterations=2)
erosion_3 = cv2.erode(img,kernel,iterations=3)
all = np.hstack((erosion_1,erosion_2,erosion_3))
cv_show('compare',all)
