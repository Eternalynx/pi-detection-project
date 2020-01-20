from __future__ import print_function
from additionals.notifications import TwilioNotifier
from additionals.utils import Conf
from imutils.video import VideoStream
from imutils.io import TempFile
from datetime import datetime
from datetime import date
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import signal
import time
import cv2
import faulthandler
import sys
from contextlib import redirect_stdout


faulthandler.enable(file = sys.stderr, all_threads = True)
faulthandler.dump_traceback(file = sys.stderr, all_threads = True)

def signal_handler(sig, frame):
    with writer:
        if writer is not None:
            writer.release()
    sys.exit(0)

def classify_frame(inputQueue, outputQueue):
    # keep looping
    while True:
        
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            gray = inputQueue.get()

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    

            # write the detections to the output queue
            
            outputQueue.put(faces)


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", default="config/config.json",
    help="Path to the  configuration json file")
ap.add_argument("-a", "--haar", default="haarcascade_frontalface_default.xml",
        help="path to haarcascade file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])
tn = TwilioNotifier(conf)

faceDetected = False
sentNotify = False

inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
faces = None



print("[INFO] starting camera")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

signal.signal(signal.SIGINT, signal_handler)
print("`ctrl + c` to exit")
face_cascade = cv2.CascadeClassifier(args["haar"])

writer = None
W = None
H = None

frameCount = 5

saveerr = sys.stderr
fsock = open('out.log','w')
sys.stderr = fsock
detectTimer = datetime.now()

p = Process(target=classify_frame, args=(inputQueue,
    outputQueue,))
p.daemon = True
p.start()

try:
    while True:
        
        frame = vs.read()
        faceDetectedPrev = faceDetected
        
        frame = imutils.resize(frame, width=400)
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        if frame is None:
                break

        
        if inputQueue.empty():
            inputQueue.put(gray)

        faceDetected  = False
        
        if not outputQueue.empty():
            faces = outputQueue.get()
            if len(faces) != 0:
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                faceDetected = True
                detectTimer = datetime.now()
            else:
                faceDetected = False
                
        
        
            
        

                
            
                
       
          

        if faceDetected  and not faceDetectedPrev and not sentNotify:
            
            startTime = datetime.now()

            fourcc = cv2.VideoWriter_fourcc('a','v','c','1')
            sentNotify = True
            
            f = open("out.log","w",encoding="utf-8")
            print("open")
            with redirect_stdout(f):
                    
                tempVideo = TempFile(basePath="./videos",ext=".mp4")
            
                
                writer = cv2.VideoWriter(tempVideo.path, fourcc, 20.0, (W, H), True)
        
            
            
        elif faceDetectedPrev:

            
            timeDiff = (datetime.now() - startTime).seconds
            if faceDetected and timeDiff > conf["open_threshold_seconds"]:
                
                if sentNotify:
                    msg = "Detecting constant motion!!!"
                    
        

                    print("send " + msg)
                    writer.release()
                    writer = None
                        
                    tn.send(msg, tempVideo)
                    sentNotify = False
                    
                    

        if (datetime.now() - detectTimer).seconds > 3 and sentNotify:
            
            sentNotify = False

            

            endTime = datetime.now()
            totalSec = (endTime - startTime).seconds
            dateDetected = date.today().strftime("%A, %B %d %Y")

            msg = "Motion was detected on {} at {} for {} " \
                    "seconds.".format(dateDetected,
                    startTime.strftime("%I:%M%p"), totalSec)
            
            print("send " + msg)
            writer.release()
            writer = None
            
            tn.send(msg, tempVideo)
            
            
        
              
        if writer is not None: 
            writer.write(frame)
            
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
finally:
    print("finally")
    if writer is not None:
        writer.release()
    sys.stderr = saveerr
    fsock.close()
    vs.stop()
    cv2.destroyAllWindows()   
    

