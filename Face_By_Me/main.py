import cv2
import numpy as np
import os
import face_recognition
#For frame count
import time
#For attendance
from datetime import datetime
#For landmarks mapping


path = 'Data/Train/Musk'
training_images = []
ClassNames = []
myList = os.listdir(path)

prev_frame_time = 0
new_frame_time = 0

for training_data in myList:
    curImg = cv2.imread(f'{path}/{training_data}')
    training_images.append(curImg)
    ClassNames.append(os.path.splitext(training_data)[0])
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4',fourcc,20,(640,480))

# ------------------- Marking encodings-------------------
def markEncodings(images):
    encodeList = []
    for img in images:
        # cv2.imshow('image', img)
        # cv2.waitKey(1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)
    return encodeList

encodedImg = markEncodings(training_images)
print("All Images Encoded")

# -------------------Attendance Function-------------------
def mark_attendance(name):
    with open('attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        print(mydatalist)
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

# ------------------- Marking faces-------------------

# cap = cv2.VideoCapture('Data/Test/pcb_test.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # ------------------- check if connected -------------------
    success, frame = cap.read()
    font = cv2.FONT_HERSHEY_COMPLEX

    # ------------------- Frame rate-------------------
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # ------------------- resizing for optimisation and changing color for processing-------------------
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    currFace = face_recognition.face_locations(imgS)
    currFacEncoddings = face_recognition.face_encodings(imgS, currFace)


    for EncodedFaces, Faces in zip(currFacEncoddings, currFace):
        matches = face_recognition.compare_faces(encodedImg, EncodedFaces)
        faceDis = face_recognition.face_distance(encodedImg, EncodedFaces)
        print(faceDis)

        matchIndex = np.argmin(faceDis)
        print(matchIndex)

        if matches[matchIndex]:
            name = ClassNames[matchIndex].upper()
            # mark_attendance(name)

            y1, x2, y2, x1 = Faces
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1, x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), font, 1, (255, 255, 255), 2)
            cv2.imshow('root', training_images[matchIndex])
            # if cv2.waitKey(1) & 0xFF == ord('g'):
            #     Img_Name = frame.copy()
            #     cv2.imshow('Captured', Img_Name)
            #     if cv2.waitKey(1) & 0xFF == ord('p'):
            #         print("You pressed p")
            #         mark_attendance(name)
            #     elif cv2.waitKey(1) & 0xFF == ord('r'):
            #         print("You pressed r")
            #         continue
            if cv2.waitKey(1) & 0xFF == ord('p'):
                Img_Name = frame.copy()
                cv2.imshow('Captured', Img_Name)
                mark_attendance(name)
                del encodedImg[matchIndex]

    # out.write(frame)
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF ==ord ('q'):
        print("You pressed q")
        break
cap.release()
cv2.destroyAllWindows()