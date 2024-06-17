import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
rpred = np.array([99])
lpred = np.array([99])
r_eye_resized = None
l_eye_resized = None

def predict_eye_state(eye):
    eye = cv2.resize(eye, (24, 24)) / 255
    eye = eye.reshape(24, 24, -1)
    eye = np.expand_dims(eye, axis=0)
    return np.argmax(model.predict(eye), axis=-1)

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        r_eye_resized = cv2.cvtColor(cv2.resize(r_eye, (24, 24)), cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        l_eye_resized = cv2.cvtColor(cv2.resize(l_eye, (24, 24)), cv2.COLOR_BGR2GRAY)

    if r_eye_resized is not None:
        rpred = predict_eye_state(r_eye_resized)
    else:
        rpred = np.array([99])

    if l_eye_resized is not None:
        lpred = predict_eye_state(l_eye_resized)
    else:
        lpred = np.array([99])

    if all(val == 0 for val in rpred) and all(val == 0 for val in lpred):
        score += 1
        if score > 20:
            try:
                sound.play()
            except:
                pass
            thicc = min(thicc + 2, 16) if thicc < 16 else max(thicc - 2, 2)
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    else:
        score = max(score - 1, 0)

    cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
