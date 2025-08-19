import cv2
import numpy as np
def mouse_callback(event,x,y,flags,userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x}, {y})')
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(img, f'({x},{y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
img=cv2.imread("C:/Users/30801/PycharmProjects/ooo/model/test.jpg")
cv2.namedWindow('Point Window')
cv2.setMouseCallback('Point Window',mouse_callback)
while True:
    cv2.imshow('Point Window',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyWindow()
