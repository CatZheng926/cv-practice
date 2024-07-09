import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('erosion.png')
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
kernel = np.ones((3,3),np.uint8)

#礼帽 = 原始-开运算
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
cv_show('morphology',tophat)

#黑帽 = 闭运算-原始
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
cv_show('name',blackhat)