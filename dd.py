from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2




def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	# A = distance.euclidean(eye[2], eye[3])

	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	
    # C = distance.euclidean(eye[4], eye[2])
    # ear = (A + B) / (2.5 * C)
	ear = (A + B) / (2.0 * C)
	return ear



thresh = 0.25
	# thresh = 0.22
	# frame_check = 20

frame_check = 25


detect = dlib.get_frontal_face_detector()
	#XML
predict = dlib.shape_predictor("D:\Downloads\shape_predictor_68_face_landmarks(2).dat")


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

from imutils import face_utils

cap=cv2.VideoCapture(0)

flag=0
while True:
	ret, frame=cap.read()
	# print(ret,frame)
		

	# frame = imutils.resize(frame, width=750)
	frame = imutils.resize(frame, width=1080)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
	subjects = detect(gray, 0)   #No upsampling in BOUNDING bOx
		
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
			
		#representing on the frame
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
			
		# cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 0), 2)
		# cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 0), 1)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			

		if ear < thresh:
			flag += 1
			#print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************Jaago ghrahak jaago!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			flag = 0
				

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		cv2.destroyAllWindows()
			
		cap.release()
		break
