from imutils.video import VideoStream
from imutils import face_utils
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import dlib
import cv2
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue

predict = None
detect = None
        

outFrame = None
lock = threading.Lock()

app = Flask(__name__)

vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src=0).start()
time.sleep(2.0)



(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


@app.route("/")
def index():
    return render_template("index.html")

def detect_eyes():
    
    global vs, outFrame, lock, predict, detect

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detect.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:

            rect = dlib.rectangle(int(x), int(y), int(x + w),
                int(y + h))

            shape = predict(gray, rect)
            shape = face_utils.shape_to_np(shape)

            lEye = shape[lStart:lEnd]
            rEye = shape[rStart:rEnd]


            lEyeHull = cv2.convexHull(lEye)
            rEyeHull = cv2.convexHull(rEye)
            cv2.drawContours(frame, [lEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rEyeHull], -1, (0, 255, 0), 1)



            
        with lock:
            outFrame = frame.copy()
        
def generate():
    global outFrame, lock

    while True:

        with lock:

            if outFrame is None:
                continue


            (flag, encodedImage) = cv2.imencode(".jpg", outFrame)

            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():

    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", default="0.0.0.0",type=str,
        help="ip address of the device")
    ap.add_argument("-o", "--port", default="8000",type=int,
        help="server port number (1024 to 65535)")
    ap.add_argument("-c", "--cascade", default="haarcascade_frontalface_default.xml",
        help = "path to face cascade")
    ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
        help="path to facial landmark predictor")
    print("[INFO] loading facial landmark predictor...")
    
   
    args = vars(ap.parse_args())
    
    detect = cv2.CascadeClassifier(args["cascade"])
    predict = dlib.shape_predictor(args["shape_predictor"])
    t = threading.Thread(target=detect_eyes, args=())
    t.daemon = True
    t.start()
    


    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)


vs.stop()