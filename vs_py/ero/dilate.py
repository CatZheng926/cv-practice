import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('erosion.png')
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv_show('name',img)
kernel = np.ones((3,3),np.uint8)
dilate_1 = cv2.dilate(img,kernel,iterations=1)
dilate_2 = cv2.dilate(img,kernel,iterations=2)
dilate_3 = cv2.dilate(img,kernel,iterations=3)
all = np.hstack((dilate_1,dilate_2,dilate_3))
cv_show('compare',all)