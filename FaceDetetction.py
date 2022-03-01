import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(image):
    face_image=image.copy()
    face_rectangle=face_cascade.detectMultiScale(face_image,scaleFactor=1.4,minNeighbors=5)
    for (x,y,w,h) in face_rectangle:
        font = cv2.FONT_HERSHEY_SIMPLEX
        draw=cv2.rectangle(face_image,(x,y),(x+w,y+h),(255,255,255),5)
        image=cv2.putText(draw, text='Face', org=((x+w+5,y+h)), fontFace=font, thickness=2,color=(0, 0, 255), lineType=cv2.LINE_AA, fontScale=1.25)

    return image

capture=cv2.VideoCapture(0)
while True:
    ret,frame = capture.read()
    if ret:
        detection=detect_face(frame)
        cv2.imshow('frame', detection)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Camera Not Working")
        break

capture.release()
cv2.destroyAllWindows()

