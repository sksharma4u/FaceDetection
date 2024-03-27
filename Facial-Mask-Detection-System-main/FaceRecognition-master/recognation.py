import cv2

facecascade=cv2.CascadeClassifier("D:\Python\OpenCV\FaceRecognition-master\HaarCascade\haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    imgGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facecascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("result",frame)

    if cv2.waitKey(33)==ord('a'):
        break

cap.release()
cv2.destroyAllWindows()

    
