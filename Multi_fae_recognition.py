#!/usr/bin/env python
# coding: utf-8

# # Multi Face Recognition

# In[ ]:


pip install face-recognition opencv-python numpy 
import face_recognition
import cv2
import os
import numpy as np

# Folder where known faces are stored
known_faces = "C:/Users/Vinoth/OneDrive - Quation Solutions Private Limited/Desktop/known_faces_images"

known_face_encoding = []  # Changed list name to singular for clarity
known_face_names = []

# Loop through each image file in the folder
for file_name in os.listdir(known_faces):
    image_path = os.path.join(known_faces, file_name)
    image = face_recognition.load_image_file(image_path)

    encodings = face_recognition.face_encodings(image)  # Correct function name
    if encodings:
        known_face_encoding.append(encodings[0])  # Store only the first encoding
        name = os.path.splitext(file_name)[0]  # Remove file extension (e.g., "person1.jpg" → "person1")
        known_face_names.append(name)

# Open webcam
video_capture = cv2.VideoCapture(0)

while True: 
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not capture an image from the webcam")
        break

    # Convert frame from BGR to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_frame) 
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding) 
        name = "Unknown"

        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding) 
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display name below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Multi-Face Recognition", frame)

    # Press 'q' to exitqq
    if cv2.waitKey(1) & 0xFF == orqd("q"):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()

