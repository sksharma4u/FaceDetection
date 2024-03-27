import tkinter as tk
#GUI
from tkinter import filedialog 
from tkinter import *
import os
import cv2
import numpy as np
from PIL import Image,ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #Model train
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time




def detect_and_predict_mask(frame, faceNet, maskNet):
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

	
	faceNet.setInput(blob)
	detections = faceNet.forward()
	
	faces = []
	locs = []
	preds = []
      
	
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		
		if confidence > 0.5:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	
	if len(faces) > 0:
		
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	
	return (locs, preds)


def detect():
    
    PATH1 = r"C:\Users\sachin sharma\Downloads\Facial-Mask-Detection-System-main\Facial-Mask-Detection-System-main\PathChanger\deploy.prototxt"
    PATH2 = r"C:\Users\sachin sharma\Downloads\Facial-Mask-Detection-System-main\Facial-Mask-Detection-System-main\PathChanger\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(PATH1, PATH2)

    
    # maskNet = load_model("C:\Users\sachin sharma\Downloads\Facial-Mask-Detection-System-main\Facial-Mask-Detection-System-main\PathChanger\mask_detector.model")
    maskNet = load_model("C:/Users/sachin sharma/Downloads/Facial-Mask-Detection-System-main/Facial-Mask-Detection-System-main/PathChanger/mask_detector.model")
    vs = VideoStream(src=0).start()

    
    while True:
        
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        
        for (box, pred) in zip(locs, preds):
            
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

           
            label = "MASK"if mask > withoutMask else "NO MASK"
            color = (0, 255, 0) if label == "MASK" else (0, 0, 255)
            
            cv2.putText(frame,"PRESS ENTER TO EXIT",(100,100),cv2.FONT_HERSHEY_COMPLEX,1.50,(0,0,255),2)
            cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        
        cv2.imshow("Frame", frame)
    
        
        if cv2.waitKey(1)==13:
            break

   
    cv2.destroyAllWindows()
    vs.stop()


def detect_from_img():
    pass
    path_to_img=path.get()
    # model = load_model('D:/OpenCV/new_mask_detection/model-090.model')

    # face_clsfr=cv2.CascadeClassifier('D:/OpenCV/new_mask_detection/haarcascade_frontalface_default.xml')

    
    cap=cv2.imread(path_to_img)
    frame=cap

    labels_dict={0:'MASK',1:'NO MASK'}
    color_dict={0:(0,255,0),1:(0,0,255)}
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:

        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]

        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],4)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],4)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_ITALIC, 1,(255,255,255),4)


    cv2.imshow('Mask Detection App',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clear():
    path.delete(first=0, last=10000)

def Live_now():
    import cv2
    cap=cv2.VideoCapture(0)
    facecascade = cv2.CascadeClassifier("C:/Users/sachin sharma/Downloads/Facial-Mask-Detection-System-main/Facial-Mask-Detection-System-main/FaceRecognition-master/HaarCascade/haarcascade_frontalface_default.xml")
    while True:
        ret,Frame=cap.read()
        cv2.putText(Frame,"PRESS ENTER TO EXIT",(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        imgGray  = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(imgGray,1.1,4)
        for(x,y,w,h) in faces:
            cv2.rectangle(Frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imshow("Result",Frame)
       
        if cv2.waitKey(1)==13:
            break
    cap.release()
    cv2.destroyAllWindows()


def browse():
    global folder_path
    global filename
    filename = filedialog.askdirectory()
    folder_path.set(filename)




window = tk.Tk()
folder_path = StringVar()
window.title("Face Mask Detection")
# window.iconbitmap('icon.ico')



width, height = window.winfo_screenwidth(), window.winfo_screenheight()

window.geometry('%dx%d+0+0' % (width,height))


image_open=Image.open('C:/Users/sachin sharma/Downloads/Facial-Mask-Detection-System-main/Facial-Mask-Detection-System-main/Tinker GUI Images/POSTER2.png')
resized=image_open.resize((width,height),Image.ANTIALIAS)
render=ImageTk.PhotoImage(resized)
img=Label(window,image=render)
img.place(x=0,y=0)

img1=Image.open('C:/Users/sachin sharma/Downloads/Facial-Mask-Detection-System-main/Facial-Mask-Detection-System-main/Tinker GUI Images/POSTER1.jpg')
res1=img1.resize((450,450),Image.ANTIALIAS)
render1=ImageTk.PhotoImage(res1)
img2=Label(window,image=render1)
img2.place(x=1060,y=170)


message = tk.Label(window, text="Live Face Mask Detection", bg="SpringGreen3", fg="black", width=50,
                   height=2, font=('times', 30, 'italic bold '))
message.place(x=200, y=20)

my_name = tk.Label(window, text="©Developed by Sanidhy Saxena", bg="Aqua", fg="black", width=60,
                   height=1, font=('times', 30, 'italic bold '))
my_name.place(x=00, y=707)

Notification = tk.Label(window, text="Live Detection", bg="snow", fg="black", width=35,
                   height=3, font=('times', 17, 'bold'))

Live_Button = tk.Button(window, text="Live Now",command= detect ,fg="white"  ,bg="purple2"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
Live_Button.place(x=140, y=310)


Exit_Button = tk.Button(window, text="Exit",command=window.quit  ,fg="white"  ,bg="purple2"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
Exit_Button.place(x=800, y=310)


window.mainloop()
