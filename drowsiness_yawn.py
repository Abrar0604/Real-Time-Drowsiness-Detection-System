# python drowniness_yawn.py --webcam webcam_index

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
import os
import requests
from datetime import datetime


yawn_timestamps = []
YAWN_TIME_WINDOW = 30  # seconds
YAWN_COUNT_THRESHOLD = 5

# ---------- EXOTEL CALL TRIGGER FUNCTION ----------
def trigger_exotel_call():
    print("[INFO] Triggering Exotel Call to Emergency Contact...")

    SID = "abrar4"
    TOKEN = "e3a544669d2acc832bff3280b239cbf21cfee645a4927962"
    FROM = "06300508010"  # Exotel verified number
    TO = "07995107476"
    EXOPHONE = "09513886363 "

    url = f"https://api.exotel.com/v1/Accounts/{SID}/Calls/connect"
    data = {
        'From': FROM,
        'To': TO,
        'CallerId': EXOPHONE,
        'CallType': 'trans'
    }

    try:
        response = requests.post(url, auth=(SID, TOKEN), data=data)
        print("[INFO] Exotel response:", response.status_code)
        print("[INFO] Exotel content:", response.text)
    except Exception as e:
        print("[ERROR] Failed to make call:", str(e))

# ---------- ALARM SOUND ----------
def sound_alarm(path):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        playsound.playsound(path)
    if alarm_status2:
        print('call')
        saying = True
        playsound.playsound(path)
        saying = False

# ---------- EAR FUNCTIONS ----------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

# ---------- ARGUMENTS ----------
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="Alert.WAV", help="path alarm .WAV file")
args = vars(ap.parse_args())

# ---------- CONSTANTS ----------
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20

alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
drowsy_start_time = None
emergency_call_triggered = False

# ---------- INITIALIZATION ----------
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# ---------- MAIN LOOP ----------
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        # Draw eyes and lips
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [shape[48:60]], -1, (0, 255, 0), 1)

        # Drowsiness logic
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Start counting drowsy time
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                else:
                    elapsed = time.time() - drowsy_start_time
                    if elapsed >= 10:
                        cv2.putText(frame, "Calling Emergency", (10, 430),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Only trigger call once, in the NEXT frame
                        if not emergency_call_triggered:
                            emergency_call_triggered = True
                            Thread(target=trigger_exotel_call).start()

            else:
                drowsy_start_time = None



        else:
            COUNTER = 0
            alarm_status = False
            drowsy_start_time = None
            emergency_call_triggered = False


    if distance > YAWN_THRESH:
        current_time = time.time()
        yawn_timestamps.append(current_time)

        # Remove yawns older than 30 seconds
        yawn_timestamps = [t for t in yawn_timestamps if current_time - t <= YAWN_TIME_WINDOW]

        cv2.putText(frame, "Yawn Alert", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Trigger alarm if yawn count exceeds threshold
        if len(yawn_timestamps) > YAWN_COUNT_THRESHOLD:
            if not alarm_status2 and not saying:
                alarm_status2 = True
                if args["alarm"] != "":
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.daemon = True
                    t.start()
    else:
        alarm_status2 = False

        # Metrics Display
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show Frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()