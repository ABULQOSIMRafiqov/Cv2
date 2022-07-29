import cv2
frameWidth = 640
frameHeight = 380
#(0)
cap = cv2.VideoCapture("resurslar/testvideo.mp4")
# cap.set(3,frameWidth)
# cap.set(4,frameHeight)

while True:
    succes,img = cap.read()
    img = cv2.resize(img,(frameWidth,frameHeight))
    cv2.imshow("video",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#img = cv2.imread("resurslar/lena.jpg")

#cv2.imshow("Lena",img)

#cv2.waitKey(0)