import mediapipe as mp
import cv2
import numpy as np
import math
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Declaration of lists
nyval = [0]
lsholyval = [0]
anglelist = [0]

#Declaration of inital variables
TNTimeCounter = 0
SlTimeCounter = 0
firstTNTime = 0
firstTNIndex = 0
firstSlTime = 0
firstSlIndex = 0
detectedTNTime = 0
detectedSlTime = 0

#Declaration of thresholds for movement
TNthresh = 0.05
Slthresh = 0.05
TNtimeThresh = 10/60
SltimeThresh = 15/60
cyclePeriod = 25/60

#Determining slouch
def isSlouch(nosey, lsholy, initnosey, initlsholy, nyval, lsholyval):
    state = False
    if (nosey >= (1 + Slthresh) * initnosey) and (lsholy >= (1 + Slthresh) * initlsholy):
        global SlTimeCounter
        SlTimeCounter += 1
        if SlTimeCounter == 1:
            print("detects")
            global firstSlTime
            firstSlTime = time.time()
            global firstSlIndex
            firstSlIndex = len(nyval) - 1
        currentTime = time.time()
        meanNosey = sum(nyval[firstSlIndex:])/len(nyval[firstSlIndex:])
        meanLsholy = sum(lsholyval[firstSlIndex:])/len(lsholyval[firstSlIndex:])

        #To check if the text neck has been held for a prolonged time
        if ((currentTime - firstSlTime) >= (SltimeThresh * 60)) and ((meanNosey >= (1 + Slthresh) * initnosey) and (meanLsholy >= (1 + Slthresh) * initlsholy)):
            state = True
            detectedSlTime = time.time()

    return state

#Function to calculate angle
def calculateAngle(a,b,c):
    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #end
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]-np.arctan2(a[1]-b[1], a[0]-b[0]))
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Phase 1 calculates initial coordinates
def phase1func(initPosRecTim, elapinitPosRecTim):
    print("into phase1func")
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark

            leftShoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
            nose = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
            leftEar = [landmarks[7].x, landmarks[7].y, landmarks[7].z]

            angle = calculateAngle(leftShoulder, nose, leftEar)
            nyval.append(landmarks[0].y)
            lsholyval.append(leftShoulder[1])
            anglelist.append(angle)
            elapinitPosRecTim = time.time() #resetting the elapsed timer

        except:
            pass

        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #cv2.imshow('Pose tracking', image)

        if (cv2.waitKey(10) & 0xFF == ord('q')) or elapinitPosRecTim - initPosRecTim > 30:
            break
    initnyval = sum(nyval[1:])/len(nyval[1:])
    initlsholyval = sum(lsholyval[1:])/len(lsholyval[1:])
    initangle = sum(anglelist[1:])/len(anglelist[1:])

    return initnyval, initlsholyval, initangle

def phase2func(initialTime, initnyval, initlsholyval, initangle):
    print("into phase2func")
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark

            leftShoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
            nose = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
            leftEar = [landmarks[7].x, landmarks[7].y, landmarks[7].z]

            angle = calculateAngle(leftShoulder, nose, leftEar)
            nyval.append(landmarks[0].y)
            lsholyval.append(leftShoulder[1])
            anglelist.append(angle)

            #To check for both text neck and slouch
            if isTextNeck(angle, initangle, anglelist) and isSlouch(landmarks[0].y, leftShoulder[1], initnyval, initlsholyval, nyval, lsholyval):
                #Code to do stretches goes here
                print("Do TN and SL stretches")
                break
            #To check for text neck
            elif isTextNeck(angle, initangle, anglelist):
                #Code to do stretches goes here
                if firstSlTime != 0:
                    print("Do TN and SL stretches")
                else:
                    print("Do TN stretches")
                break
            #To check for slouch
            elif isSlouch(landmarks[0].y, leftShoulder[1], initnyval, initlsholyval, nyval, lsholyval):
                #Code to do stretches goes here
                if firstTNTime != 0:
                    print("Do TN and SL stretches")
                else:
                    print("Do SL stretches")
                break

            currentTime = time.time()
            #To check if the cycle is over
            if (currentTime - initialTime) >= cyclePeriod * 60:
                #User has sat in good posture for 25 min and now needs to be rewarded and should move
                print("Great job! Move around a bit and get back to work!")
                break

        except:
            pass

        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #cv2.imshow('Pose tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

#Final run through function
def runThrough():
    print("recording initial times")
    initialTime = time.time()
    elapsedTime = time.time()
    print("going into phase1func")
    initnyval, initlsholyval, initangle = phase1func(initialTime, elapsedTime)
    print(initnyval, initlsholyval, initangle)
    print("again recording initial time")
    initialTime = time.time()
    print("going into phase2func")
    phase2func(initialTime, initnyval, initlsholyval, initangle)
    print("finished phase2func")
    time.sleep(5)
    print("resetting variables")

    global nyval
    nyval = [0]

    global lsholyval
    lsholyval = [0]

    global anglelist
    anglelist = [0]

    global TNTimeCounter
    TNTimeCounter = 0

    global SlTimeCounter
    SlTimeCounter = 0

    global firstTNTime
    firstTNTime = 0

    global firstTNIndex
    firstTNIndex = 0

    global firstSlTime
    firstSlTime = 0

    global firstSlIndex
    firstSlIndex = 0

    global detectedTNTime
    detectedTNTime = 0

    global detectedSlTime
    detectedSlTime = 0

    print("starting again")
    runThrough()

cap = cv2.VideoCapture(1)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    runThrough()