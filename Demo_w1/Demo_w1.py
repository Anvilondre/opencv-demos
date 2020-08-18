import cv2
import numpy as np

cap = cv2.VideoCapture(0)
hb_img = cv2.imread("data/hb.jpg")
x = y = w = h = 1
area_min = 40_000


def processContours(img, imgContour):
    global area_min
    global x, y, w, h
    contours, *_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_min:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)


while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgContour = img.copy()
    imgReplaced = imgContour.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 2)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, 100, 255)
    imgDil = cv2.dilate(imgCanny, (5, 5), iterations=1)
    processContours(imgDil, imgContour)
    hb_resized = cv2.resize(hb_img, (w, h))
    imgReplaced[y: y + h, x: x + w] = hb_resized
    imgStack = np.hstack((imgContour, imgReplaced))
    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
