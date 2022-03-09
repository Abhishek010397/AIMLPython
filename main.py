import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import datetime
import face_recognition
import dlib
import pymongo
from flask import Flask,Response,render_template

app = Flask(__name__)

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
    human_image=image
    human_rectangle=human_cascade.detectMultiScale(human_image,scaleFactor=1.02,minNeighbors=4)
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
            return ("Access Denied")

def gen_frames():
    capture=cv2.VideoCapture("rtsp://admin:cctv@123@103.82.81.99:554/")
    count = 1
    while True:
        ret,frame = capture.read()
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret:
            value=None
            detection,val=detect_human(grayFrame,value)
            print(val)
            if val == True:
                font = cv2.FONT_HERSHEY_SIMPLEX
                validate_face=face_validation(frame)
                print(validate_face)
                if validate_face == "Access Granted":
                    if count == 1:
                        now = datetime.datetime.now()
                        now = now.strftime("%H:%M:%S")
                        cv2.putText(detection, text=validate_face, org=(35, 425), fontFace=font, thickness=2,
                                    color=(255, 0, 0), lineType=cv2.LINE_AA, fontScale=1.25)
                        cv2.imwrite(now + 'frame.jpeg', detection)
                        ret,frame = cv2.imencode('.jpeg', detection)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
                        count=count+1
                    if count > 1:
                        ret,frame = cv2.imencode('.jpeg', detection)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
                        count=count+1
                if validate_face == "Access Denied":
                    if count == 1:
                        now = datetime.datetime.now()
                        now = now.strftime("%H:%M:%S")
                        cv2.putText(detection, text=validate_face, org=(35, 425), fontFace=font, thickness=2,
                                    color=(255, 0, 0), lineType=cv2.LINE_AA, fontScale=1.25)
                        cv2.imwrite(now + 'frame.jpeg', detection)
                        ret,frame = cv2.imencode('.jpeg', detection)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
                        count=count+1
                    if count> 1 :
                        ret,frame = cv2.imencode('.jpeg', detection)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
                        count=count + 1
            if val == False:
                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, frame = cv2.imencode('.jpeg', grayFrame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
        else:
            print("Can't Open Camera")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



