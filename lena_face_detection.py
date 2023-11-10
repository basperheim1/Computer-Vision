import cv2 as cv

lena = cv.imread('Resources/lena.png')

gray_lena = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = haar_cascade.detectMultiScale(gray_lena, scaleFactor=1.3, minNeighbors=9)

for (x, y, w, h) in faces:
    cv.rectangle(lena, (x, y), (x+w, y+h), (255, 0, 0))

cv.imshow("Lena's Detected Face", lena)
cv.waitKey(0)