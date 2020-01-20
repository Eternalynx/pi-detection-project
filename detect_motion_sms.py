from __future__ import print_function
from additionals.notifications import TwilioNotifier
from additionals.utils import Conf
from additionals.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from imutils.io import TempFile
from datetime import datetime
from datetime import date
import numpy as np
import argparse
import imutils
import signal
import time
import cv2
import sys
import faulthandler
from contextlib import redirect_stdout


def signal_handler(sig, frame):
    with writer:
        if writer is not None:
            writer.release()
    sys.exit(0)


faulthandler.enable(file = sys.stderr, all_threads = True)
faulthandler.dump_traceback(file = sys.stderr, all_threads = True)
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", default="config/config.json",
    help="Path to the configuration json file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])
tn = TwilioNotifier(conf)

motionDetected = False
sentNotify = False

print("[INFO] starting camera")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()

time.sleep(2.0)

signal.signal(signal.SIGINT, signal_handler)
print("`ctrl + c` to exit")

writer = None
W = None
H = None

md = SingleMotionDetector(weight=0.1)
total = 0
frameCount = 5

saveerr = sys.stderr
fsock = open('out.log','w')
sys.stderr = fsock
detectTimer = datetime.now()

while True:
    
    frame = vs.read()
    motionDetectedPrev = motionDetected


    
    if frame is None:
            break

    
    frame = imutils.resize(frame, width=400)
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)


    if W is None or H is None:
        (H, W) = frame.shape[:2]

    
    
    if total > frameCount:
        detMotion = md.detect(gray)
        
        if detMotion is not None:
            
            detectTimer = datetime.now()
            
            (thresh, (minX, minY, maxX, maxY)) = detMotion
            if maxX-minX>10 and maxY-minY >10:
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),(0, 0, 255), 2)
                #print("adding rect ",minX," ",maxX," ",minY," ",maxY)
                motionDetected = True
            else:
                motionDetected = False
        else:
            motionDetected  = False
      

    if motionDetected and not motionDetectedPrev and not sentNotify:
        
        startTime = datetime.now()

        fourcc = cv2.VideoWriter_fourcc('a','v','c','1')
        sentNotify = True
        
        f = open("out.log","w",encoding="utf-8")
        print("open")
        with redirect_stdout(f):
                
            tempVideo = TempFile(basePath="./videos",ext=".mp4")
        
        
            writer = cv2.VideoWriter(tempVideo.path, fourcc, 20.0, (W, H), True)
    
        print("test")
        
    elif motionDetectedPrev:

        
        timeDiff = (datetime.now() - startTime).seconds

        if motionDetected and timeDiff > conf["open_threshold_seconds"]:
            
            if sentNotify:
                msg = "Detecting constant motion!!!"

                writer.release()
            
                writer = None

                tn.send(msg, tempVideo)
                print("send " + msg)
                sentNotify = False


    if (datetime.now() - detectTimer).seconds > 2 and sentNotify:
        
        sentNotify = False

        

        endTime = datetime.now()
        totalSec = (endTime - startTime).seconds
        dateDetected = date.today().strftime("%A, %B %d %Y")

        msg = "Motion was detected on {} at {} for {} " \
                "seconds.".format(dateDetected,
                startTime.strftime("%I:%M%p"), totalSec)

        writer.release()
        writer = None

        tn.send(msg, tempVideo)
        print("send " + msg)
            
    if writer is not None:
        writer.write(frame)
    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    md.update(gray)
    total += 1

if writer is not None:
    writer.release()
sys.stderr = saveerr
fsock.close()
cv2.destroyAllWindows()
vs.stop()