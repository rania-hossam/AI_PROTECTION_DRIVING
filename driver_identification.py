import cv2
import face_recognition

# Load images of known drivers
driver1_image = face_recognition.load_image_file("C:\\Users\\rania\Downloads\\AI_DRIVER_SYSTEM\\IMAGE.jpg")
driver2_image = face_recognition.load_image_file("C:\\Users\\rania\Downloads\\AI_DRIVER_SYSTEM\\image2.jpg")

# Get facial encodings of known drivers
driver1_encoding = face_recognition.face_encodings(driver1_image)[0]
driver2_encoding = face_recognition.face_encodings(driver2_image)[0]

# Create a list of known drivers and their encodings
known_drivers = [
    {"name": "Driver 1", "encoding": driver1_encoding},
    {"name": "Driver 2", "encoding": driver2_encoding}
]

# Start the video stream from the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for any known drivers
        matches = face_recognition.compare_faces([driver["encoding"] for driver in known_drivers], face_encoding)

        name = "Unknown"
        for i in range(len(matches)):
            if matches[i]:
                name = known_drivers[i]["name"]
                break

        # Draw a box around the face and put the name of the driver
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Wait for the user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
