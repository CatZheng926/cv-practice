import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('erosion.png')
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#cv_show('morphology',img)
'''
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv_show('morphology',opening)
'''
'''
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv_show('morphology',closing)
'''
kernel = np.ones((3,3),np.uint8)
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
cv_show('morphology',gradient)