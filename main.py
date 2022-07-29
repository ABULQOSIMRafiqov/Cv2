import cv2


img = cv2.imread('resurslar/star.jpg')

print(img.shape)
width, height = 1000, 1000
imgResize = cv2.resize(img,(width,height))
print(imgResize.shape)


imgCropped = img[500:2880,0:1800]
imgCropResize = cv2.resize(imgCropped,(img.shape[1],img.shape[0]))




cv2.imshow("Star",img)

#cv2.imshow("Star Resized",imgResize)
cv2.imshow("Star Cropped",imgCropped)
cv2.imshow("Star Cropped, Resized",imgResize)
cv2.waitKey(0)
