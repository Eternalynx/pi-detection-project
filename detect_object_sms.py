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
import sys
from contextlib import redirect_stdout

def signal_handler(sig, frame):
    with writer:
        if writer is not None:
            writer.release()
    sys.exit(0)


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", default="config/config.json",
    help="Path to the  configuration json file")
ap.add_argument("-t", "--threshold", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-o", "--object", default="bottle",
    help="object to follow")
args = vars(ap.parse_args())

conf = Conf(args["conf"])
tn = TwilioNotifier(conf)
obj = args["object"]
objectDetected = False
sentNotify = False

def classify_frame(net, inputQueue, outputQueue):
    while True:
        if not inputQueue.empty():

            frame = inputQueue.get()
            frame = cv2.resize(frame, (300, 300))
            blob = cv2.dnn.blobFromImage(frame, 0.007843,
                (300, 300), 127.5)

            net.setInput(blob)
            detections = net.forward()

            outputQueue.put(detections)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,
    outputQueue,))
p.daemon = True
p.start()


print("[INFO] starting camera")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

signal.signal(signal.SIGINT, signal_handler)
print("`ctrl + c` to exit")

writer = None
W = None
H = None

frameCount = 5

saveerr = sys.stderr
fsock = open('out.log','w')
sys.stderr = fsock
detectTimer = datetime.now()

p = Process(target=classify_frame, args=(net, inputQueue,
    outputQueue))
p.daemon = True
p.start()
detectFlag = False
while True:
    
    frame = vs.read()
    objectDetectedPrev = objectDetected
    
    frame = imutils.resize(frame, width=400)



    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    if frame is None:
            break

    
    if inputQueue.empty():
        inputQueue.put(frame)

    objectDetected  = False
    if not outputQueue.empty():
        detections = outputQueue.get()
        
    detectFlag = False
    if detections is not None:
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]


            if confidence < args["threshold"]:
                continue

            idx = int(detections[0, 0, i, 1])
        
            if CLASSES[idx] == obj:
                detectFlag = True
                
                dims = np.array([W, H, W, H])
                box = detections[0, 0, i, 3:7] * dims
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)    
        
        
        
    if detectFlag:
        objectDetected = True
        detectTimer = datetime.now()
    else:
        objectDetected = False
                     
        
    if objectDetected and sentNotify:
        sentNotify = False
        writer.release()
        writer = None
        tempVideo.cleanup()
        tempVideo = None
      

    if not objectDetected  and objectDetectedPrev and not sentNotify:
        
        startTime = datetime.now()

        fourcc = cv2.VideoWriter_fourcc('a','v','c','1')
        sentNotify = True
        
        f = open("out.log","w",encoding="utf-8")
        print("open")
        with redirect_stdout(f):
                
            tempVideo = TempFile(basePath="./videos",ext=".mp4")
        
        
            writer = cv2.VideoWriter(tempVideo.path, fourcc, 20.0, (W, H), True)
    

    if (datetime.now() - detectTimer).seconds > 5 and sentNotify:
        
        sentNotify = False

        

        endTime = datetime.now()
        totalSec = (endTime - startTime).seconds
        dateDetected = date.today().strftime("%A, %B %d %Y")

        msg = "Object not detected on {} at {} for {} " \
                "seconds.".format(dateDetected,
                startTime.strftime("%I:%M%p"), totalSec)
        
       
        writer.release()
        writer = None
        
        tn.send(msg, tempVideo)
        
    
            
    if writer is not None:
       
        writer.write(frame)
    
    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    
    

if writer is not None:
    writer.release()
sys.stderr = saveerr
fsock.close()
cv2.destroyAllWindows()
vs.stop()
