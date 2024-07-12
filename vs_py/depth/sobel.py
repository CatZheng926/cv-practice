import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('erosion.png')
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
titles = [sobel_x,sobel_y]
imgs = [sobel_x,sobel_y]
plt.figure(figsize=(10, 5))         #调整图像显示的大小
plt.subplot(1,2,1),plt.imshow(imgs[0],'gray'),plt.title(titles[0])
plt.subplot(1,2,2),plt.imshow(imgs[1],'gray'),plt.title(titles[1])
plt.suptitle('Sobel Edge Detection')
plt.show()


