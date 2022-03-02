import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import datetime
import face_recognition
import dlib
import pymongo


try:
    client = pymongo.MongoClient(host='hostname', port=27017, username='root', password='pass', authSource="admin")
    print("okay connection!!")
    db = client["users_db"]
    col = db.webcam_recognize
    transaction_db = db.transaction_table
except:
    print("error in connection")


human_cascade=cv2.CascadeClassifier('haarcascade_fullbody.xml')

def detect_human(image,value):
    human_image=image.copy()
    human_rectangle=human_cascade.detectMultiScale(human_image,scaleFactor=1.01,minNeighbors=5)
    print(human_rectangle)
    if type(human_rectangle) == numpy.ndarray and type(human_rectangle) == numpy.ndarray:
        value=True
    for (x,y,w,h) in human_rectangle:
        font = cv2.FONT_HERSHEY_SIMPLEX
        image=cv2.putText(human_image, text='Person', org=((x,y+h)), fontFace=font, thickness=2,color=(0, 0, 255), lineType=cv2.LINE_AA, fontScale=1.25)
    return image,value

def face_validation(frame):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    facesCurFrame = face_recognition.face_locations(rgb_frame)
    encodesCurFrame = face_recognition.face_encodings(
        rgb_frame, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        name = ''
        id = 0
        for record in col.find({}):
            flag = 0
            for k, v in record.items():
                if (k.lower() == "pixelvalue"):
                    encodeListKnown = tuple(v[0])
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = name.upper()
                        flag = 1
                        return ("Access Granted")
        if (flag == 0):
            return ("ACCESS DENIED!!")


capture=cv2.VideoCapture(0)
while True:
    ret,frame = capture.read()
    if ret:
        value = None
        detection,val=detect_human(frame,value)
        print(val)
        if val == True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            validate_face=face_validation(frame)
            cv2.putText(detection, text=validate_face, org=(35, 425), fontFace=font, thickness=2,
                        color=(255, 0, 0), lineType=cv2.LINE_AA, fontScale=1.25)
            cv2.imshow('frame',detection)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Camera Not Working")
        break

capture.release()
cv2.destroyAllWindows()

