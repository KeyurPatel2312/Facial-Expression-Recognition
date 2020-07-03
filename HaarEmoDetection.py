'''
Using HAAR Cascade
'''


import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()

	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]
	gray = cv2.cvtColor(rgb_small_frame,cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	face2=[]
	for (x,y,w,h) in faces:
		x*=4
		y*=4
		w*=4
		h*=4
		face2.append((x,y,w,h))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		print(str((x,y,w,h))+" "+str(detected_face.shape))
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		 
		img_pixels /= 255
		 
		predictions = model.predict(img_pixels)
		 
		#find max indexed array
		max_index = np.argmax(predictions[0])
		 
		emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
		emotion = emotions[max_index]
		 
		cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
				
	# Display the resulting image
	cv2.imshow('Video', frame)
	c=1
	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		print((x,y,w,h))
		for (x,y,w,h) in face2:
			detected_face = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			 
			img_pixels /= 255
			 
			predictions = model.predict(img_pixels)
			 
			#find max indexed array
			max_index = np.argmax(predictions[0])
			 
			emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
			emotion = emotions[max_index]
			cv2.imwrite("C:\\Users\\Keyur Patel\\Desktop\\miniproject\\emotions"+"/"+str(c)+"Haar"+emotion+".jpg",frame[y:y+h,x:x+w])
			show_img=frame[y:y+h,x:x+w]
			#cv2.putText(show_img, emotion, (int(x/2), int(y/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			cv2.imshow(emotion,show_img)
			c=c+1
			cv2.waitKey(0)
		break
		"""
		detected_face = cv2.cvtColor(cv2.imread("emotion.jpg"), cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		 
		img_pixels /= 255
		 
		predictions = model.predict(img_pixels)
		 
		#find max indexed array
		max_index = np.argmax(predictions[0])
		 
		emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
		emotion = emotions[max_index]
		show_img=frame[y:y+h,x:x+w]
		cv2.putText(show_img, emotion, (int(x/2), int(y/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		cv2.imshow("Emotion",show_img)
		cv2.waitKey(0)
		break
		"""

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()