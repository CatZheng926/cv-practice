import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('cat.jpg')
couple = cv2.imread('couple.jpg')
def cv_show(name,img):#name是窗口名字
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#cv_show('image',img)#这里我装了个pip install opencv-contrib-python解决了之前OpenCVcv缺少依赖的问题
#print(img.shape)#展示图片尺寸
#img1 = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)#这里是显示了这张图片的灰度图
#cv_show('image',img1)

#7.5
#这里是获取视频然后进行一些操作并输出
'''
vc = cv2.VideoCapture('Apex.mp4')
if vc.isOpened():#检查是否可以打开
    open#,frame = vc.read()        我觉得这里把frame=vc.read()去掉可以解决第一帧丢失的问题
else:
    open = False
while open:
    ret,frame = vc.read()
    if frame is None :
        break
    if ret == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#灰度处理
        cv2.imshow('result',gray)
        if cv2.waitKey(10)&0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()
'''

#只保留BGR中的某一项
'''
cat = img.copy()
#cat[:,:,1] = 0
cat[:,:,2] = 0
cv_show('Bcat',cat)
'''

#这里是给图片进行边框延展，注意用plt.show输出时是RGB，cv读取时BGR
'''
top_size,bottom_size,left_size,right_size = (50,50,50,50)
#frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
constant = cv2.copyMakeBorder(img_rgb,50,50,50,50,cv2.BORDER_CONSTANT,value = 0)
plt.imshow(constant),plt.title('constant')
plt.show()
#cv_show('cat',img)
'''

#对图像大小进行裁剪，或者按比例进行缩放
'''
#print(img[:2,:2,1])
cat = img[:,:,1]
cv_show('Bcat',cat)
#resize(W,H)  裁剪大小，长宽比例缩放
#shape = (H,W)
'''

#图像融合
'''
couple = cv2.resize(couple,(444,294))
#cv_show('name',couple)
fix = cv2.addWeighted(couple,0.5,img,0.5,0)
cv_show('fix',fix)
'''
7.6
'''
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray= cv2.cvtColor(img_gray,cv2.COLOR_RGB2GRAY)
#cv_show('name',img_gray)#图像阈值
ret,thresh1 = cv2.threshold (img_gray,127,255,cv2.THRESH_BINARY)     #0或者最大  二值模式
ret,thresh2 = cv2.threshold (img_gray,127,255,cv2.THRESH_BINARY_INV) #与1相反
ret,thresh3 = cv2.threshold (img_gray,127,255,cv2.THRESH_TRUNC)      #大于取阈值，否则不变
ret,thresh4 = cv2.threshold (img_gray,127,255,cv2.THRESH_TOZERO)     #大于不变,否则取0
ret,thresh5 = cv2.threshold (img_gray,127,255,cv2.THRESH_TOZERO_INV) #与4相反
titles = [img_gray,thresh1,thresh2,thresh3,thresh4,thresh5]
images = [img_gray,thresh1,thresh2,thresh3,thresh4,thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
plt.show()
'''

'''
#内核越大，图片整体越模糊
#均值滤波
#简单的平均卷积操作
blur = cv2.blur(img,(10,10))
cv_show('name',blur)

#方框滤波
#基本和均值一样，可以选择归一化
box = cv2.boxFilter(img,-1,(3,3),normalize=True)
cv_show('box',box)

#高斯滤波
#更重视中间的
aussian = cv2.GaussianBlur(img,(5,5),1)  #1是方差
cv_show('aussian',aussian)

#中值滤波
#排序中位数做处理结果
median = cv2.medianBlur(img,5)
cv_show('median',median)

#展示所有结果  横向排列
res = np.hstack((blur,aussian,median))
cv_show('compare',res)
'''

'''
7.8
#腐蚀操作
#内核(kernel越大腐蚀得越快)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,img = cv2.threshold (gray,127,255,cv2.THRESH_BINARY) 
#上面两行是把三通道图转换为二值图
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations=2)
cv_show('erosion',erosion)
'''
'''
#膨胀操作
kernel = np.ones((3,3),np.uint8)
dilate = cv2.dilate(img,kernel,iterations=1)
cv_show('dilate',dilate)
'''



