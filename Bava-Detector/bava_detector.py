import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')
prof_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_profileface.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
    #eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    profiles = prof_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for(x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 90:
            print(id_)
        color = (255, 0, 0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    for(x, y, w, h) in profiles:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)

        color = (255, 0, 0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    """for(x, y, w, h) in eyes:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)

        color = (255, 0, 0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)"""
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
