import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import pyttsx3

#include face detection trained model
face_brain=cv2.CascadeClassifier("face-ai-brain.xml")
engine = pyttsx3.init() #intitiatig pyttsx3 module
def talk_function(text): #Define talk funtion
	engine.say(text)
	engine.runAndWait()
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female | 0 for male
np.set_printoptions(suppress=True) 
model = tensorflow.keras.models.load_model('keras_model.h5') #load model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)  #create an empty numpy (224 X 244)
video_device = cv2.VideoCapture(0)


while True:
	ret,frame = video_device.read()
	font=cv2.FONT_HERSHEY_SIMPLEX
	gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_brain.detectMultiScale(gray_frame,1.2,5)
	print(frame)
	resized_frame = cv2.resize(frame,(224 , 224))
	image_array = np.asarray(resized_frame)
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
	data[0] = normalized_image_array
	prediction = model.predict(data)
	print(prediction)
	in_lectures = prediction[0][0] 
	not_in_lectures = prediction[0][1]
	#Initiating Logic
	if in_lectures > not_in_lectures:
		for(x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
			cv2.putText(frame,'face detected',(x+w,y+h),font,1,(255,0,0),2,cv2.LINE_AA)
		print("You are attending lectures")
		talk_function("You attending lectures")
	else:
		print("You are not in lectures")
		talk_function("You are not in lectures")
	cv2.imshow("Thareejan Screen", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break 

video_device.release()
video_device.destroyAllWindows()