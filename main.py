# # Demo Trackbar
# import cv2
# import numpy as np
#
# def nothing(x):
#     pass
#
#
# img = numpy.zeros((250, 500, 3), numpy.uint8)
# cv2.namedWindow('image')
#
# # creating trackbars for red color change
# cv2.createTrackbar('R', 'image', 8, 179, nothing)
#
# # creating trackbars for Green color change
# cv2.createTrackbar('G', 'image', 255, 255, nothing)
#
# # creating trackbars for Blue color change
# cv2.createTrackbar('B', 'image', 255, 255, nothing)
#
# while (True):
#     # show image
#     cv2.imshow('image', img)
#
#     # for button pressing and changing
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#
#     # get current positions of all Three trackbars
#     r = cv2.getTrackbarPos('R', 'image')
#     g = cv2.getTrackbarPos('G', 'image')
#     b = cv2.getTrackbarPos('B', 'image')
#
#     # display color mixture
#     img[:] = [b, g, r]

# close the window
#cv2.destroyAllWindows()
# import cv2
# import numpy

# loading the image into a numpy array
# img = cv2.imread('cards.jpg')
# color = (255,0,0)
# thickness = 10
# # start_point = (120,50)
# # end_point = (250,250)
# center_coordinates=(120,100)
# axeslength = (100,50)
# angle = 30
# startangle = 0
# endangle = 360
# #radius = 20
# img = cv2.ellipse(img, center_coordinates,axeslength, angle, startangle, endangle, color, thickness)
#
# cv2.imshow('frame', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# img = cv.imread(cv.samples.findFile("cards.jpg"))
# if img is None:
#     sys.exit("Could not read the image.")
# cv.imshow("Display window", img)
# k = cv.waitKey(0)
# if k == ord("s"):
#     cv.imwrite("starry_night.png", img)

# import cv2

# import sys
# img = cv2.imread('cards.jpg',-2)
# cv2.imshow('frame', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
#
# cap = cv2.VideoCapture(0)
#
# if(cap.isOpened() == False):
#     print("Camera could not open")
#
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# #video coded
# video_cod =  cv2.VideoWriter_fourcc(*'XVID')
# video_output = cv2.VideoWriter('Captured_Video.MP4', video_cod, 30, (frame_width, frame_height))
# # abc = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if ret == True:
#         video_output.write(frame)
#         cv2.imshow('frame', frame)
#
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
# video_output.release()
# cv2.destroyAllWindows()
# print("The video was saved successfully")

# import cv2
# cap = cv2.VideoCapture("Captured_Video.MP4")
# while True:
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


# import numpy as np
# import cv2
#
# img = cv2.imread('cards.jpg')
# px = img[100,100]
# print(px)
# blue= img[100,100,0]
# print(blue)
# img[100,100]=[255,255,255]
# print(img[100,100])

# import numpy as np
# import cv2
# img_file = 'cards.jpg'
# img = cv2.imread(img_file, cv2.IMREAD_COLOR)
# alpha = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
# gray_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
#
# print('RGB shape : ', img.shape)
# print('ARGB shape : ', alpha.shape)
# print('Gray Scale shape : ', gray_img.shape)

#
# import cv2
# img_file = 'cards.jpg'
# img_raw = cv2.imread(img_file)
# roi = cv2.selectROI(img_raw)
# print(roi)
# #cropping selected ROI from the raw(given) image
#
# roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]) :int(roi[0]+roi[2])]
# cv2.imshow("RoI image", roi_cropped)
# cv2.imwrite("cropped.jpeg", roi_cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# img_file = 'cards.jpg'
# img = cv2.imread(img_file)
#
# g,b,r = cv2.split(img)
# cv2.imshow("Green part of the image", g)
# cv2.imshow("Red part of the image", r)
# cv2.imshow("Blue part of the image", b)
#
# img1 = cv2.merge((g,b,r))
# cv2.imshow("Image after of three colors", img1)
# cv2.waitKey(0)


# import cv2
# img_file = 'cards.jpg'
# img = cv2.imread(img_file)
# color_change = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
# cv2.imshow("Image after of three colors", color_change)
#
#
# cv2.waitKey(0)

# import cv2
# src1 = cv2.imread('cards.jpg', cv2.IMREAD_COLOR)
# src2 = cv2.imread('resurslar/book.png', cv2.IMREAD_COLOR)
# img1 = cv2.resize(src1, (800,600))
# img2 = cv2.resize(src2, (800,600))
# blended_img = cv2.addWeighted(img1,0.5,img2, 1, 1)
#
# cv2.imshow("Blender/additive image", blended_img)
#
#
# cv2.waitKey(0)

# import cv2
# import numpy as np
# img = cv2.imread('cards.jpg')
# k_sharped = np.array([[1, 1, 1],
#                      [1,-9,1],
#                      [1,1,1]])
# sharpened = cv2.filter2D(img, -1, k_sharped)
# cv2.imshow('Original image',img)
# cv2.imshow('filtered Image',sharpened)
# cv2.waitKey(0)

#
# import cv2
# import numpy as np
# img = cv2.imread('cards.jpg',cv2.IMREAD_GRAYSCALE)
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# cannyimage = cv2.Canny(img, 50, 100)
# cv2.imshow('Original image',img)
# cv2.imshow('Threshhold',thresh)
# cv2.imshow('Canny image',cannyimage)
# cv2.waitKey(0)


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('shapes.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# i=0
# for contour in contours:
#     if i==0:
#         i=1
#         continue
#
#     appox = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
#     cv2.drawContours(img, [contour], 0 , (255, 0, 255),5)
#
#     M = cv2.moments(contour)
#     if M['m00'] != 0.0:
#         x = int(M['m10']/M['m00'])
#         y = int(M['m01'] / M['m00'])
#
#     if len(appox) == 3:
#         cv2.putText(img, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#     elif len(appox) == 4:
#         cv2.putText(img, 'Quadrilaterla', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#     elif len(appox) == 6:
#         cv2.putText(img, 'Hexagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#     else:
#         cv2.putText(img, 'circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#
# cv2.imshow('shapes',img)
# cv2.waitKey(0)

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # reading image
# img = cv2.imread('shapes.png')
#
# # converting image into grayscale image
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # setting threshold of gray image
# _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
# # using a findContours() function
# contours, _ = cv2.findContours(
#     threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# i = 0
#
# # list for storing names of shapes
# for contour in contours:
#
#     # here we are ignoring first counter because
#     # findcontour function detects whole image as shape
#     if i == 0:
#         i = 1
#         continue
#
#     # cv2.approxPloyDP() function to approximate the shape
#     approx = cv2.approxPolyDP(
#         contour, 0.01 * cv2.arcLength(contour, True), True)
#
#     # using drawContours() function
#     cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
#
#     # finding center point of shape
#     M = cv2.moments(contour)
#     if M['m00'] != 0.0:
#         x = int(M['m10'] / M['m00'])
#         y = int(M['m01'] / M['m00'])
#
#     # putting shape name at center of each shape
#     if len(approx) == 3:
#         cv2.putText(img, 'Triangle', (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#     elif len(approx) == 4:
#         cv2.putText(img, 'Quadrilateral', (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#     elif len(approx) == 5:
#         cv2.putText(img, 'Pentagon', (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#     elif len(approx) == 6:
#         cv2.putText(img, 'Hexagon', (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#     else:
#         cv2.putText(img, 'circle', (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
# # displaying the image after drawing contours
# cv2.imshow('shapes', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# img = cv2.imread('shapes.png')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # lower_blue = np.array([0,50,50])
# # upper_blue = np.array([140,255,255])
# #
# # lower_green = np.array([40,40,40])
# # upper_green = np.array([70,255,255])
#
#
# lower_yellow = np.array([10,100,20])
# upper_yellow= np.array([25,255,255])
# mask_blue = cv2.inRange(hsv, lower_yellow, upper_yellow)
# #mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
# res = cv2.bitwise_and(img, img, mask=mask_blue)
# cv2.imshow('res',res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('cards.jpg', cv2.IMREAD_COLOR)
# img1 = img.copy()
# mask = np.zeros((100,300,3))
# pos = (200,200)
# var = img1[200:(200+mask.shape[0]), 200:(200+mask.shape[1])]=mask
# cv2.imshow("colring", img1)
# cv2.waitKey(0)