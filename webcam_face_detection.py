import cv2 as cv


haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

facecam = cv.VideoCapture(0)


while True:
    is_true, frame = facecam.read()

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv.imshow('Webcam Detected Faces', frame)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break

facecam.release()
cv.destroyAllWindows()


