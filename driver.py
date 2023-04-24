import cv2
import os

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Load the list of driver names
driver_names = os.listdir('drivers')

# Initialize the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y+h, x:x+w]

        # Recognize the face using the pre-trained model
        label, confidence = recognizer.predict(face)

        # Check if the confidence is high enough to consider the recognition result
        if confidence < 100:
            driver_name = driver_names[label]
            print('Driver:', driver_name)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with the detected faces
    cv2.imshow('frame', frame)

    # Check for user input to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources used by the video capture device
cap.release()
cv2.destroyAllWindows()
