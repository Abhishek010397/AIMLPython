import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import datetime


face_cascade=cv2.CascadeClassifier('haarcascade_fullbody.xml')

def detect_face(image,value):
    face_image=image.copy()
    face_rectangle=face_cascade.detectMultiScale(face_image,scaleFactor=1.01,minNeighbors=5)
    print(face_rectangle)
    if type(face_rectangle) == numpy.ndarray and type(face_rectangle) == numpy.ndarray:
        value=True
    for (x,y,w,h) in face_rectangle:
        font = cv2.FONT_HERSHEY_SIMPLEX
        image=cv2.putText(face_image, text='Person', org=((x,y+h)), fontFace=font, thickness=2,color=(0, 0, 255), lineType=cv2.LINE_AA, fontScale=1.25)
    return image,value

capture=cv2.VideoCapture(0)
while True:
    ret,frame = capture.read()
    if ret:
        value = None
        detection,val=detect_face(frame,value)
        print(val)
        if val == True:
            now = datetime.datetime.now()
            now = now.strftime("%H:%M:%S")
            cv2.imwrite(now+"frame.jpeg", frame)
            cv2.imshow('frame', detection)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Camera Not Working")
        break

capture.release()
cv2.destroyAllWindows()

