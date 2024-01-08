import cv2
import cv2.data
import streamlit as st
import os
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(selected_color, min_neighbors, scale_factor):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Create the directory to save images if it doesn't exist
    os.makedirs('faces_file', exist_ok=True)

    capture_count = 0  # Counter for captured images
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()

        if not ret:
            print('Frame not captured')
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Draw rectangles around detected faces using the selected color
        for (x, y, w, h) in faces:
            b, g, r = int(selected_color[5:7], 16), int(selected_color[3:5], 16), int(selected_color[1:3], 16)
            bgr_color = (b, g, r)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, thickness=2)
            # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Display the frame with rectangles using Streamlit

        img_name = f'faces_file/captured_image_{capture_count}.jpg'
        cv2.imwrite(img_name, frame)

        capture_count += 1
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()

# Streamlit app
def main():
    st.title('Face Detection and capturing ')

    st.write('* select the rectangle face detection color from the pick a color mini box')
    st.write('* control the minNeighbor slider to affect the accuracy of face detection')
    st.write('* control the scaleFactor slider to adjust the sensitivity for detecting faces')
    st.write('* click the Start Face Detection buttom to get started after youve adjusted the parametres')
    st.write('* to stop the detection and capturing process click q')
    st.write('* the captured images can be found in the file named faces_file ')
    # Display color picker for rectangles
    selected_color = st.color_picker('Pick a color', '#ff6347')  # Default color: Tomato

    # Adjust minNeighbors parameter
    min_neighbors = st.slider('minNeighbors', min_value=1, max_value=10, value=5)

    # Adjust scaleFactor parameter
    scale_factor = st.slider('scaleFactor', min_value=1.1, max_value=2.0, value=1.2, step=0.1)

    # Button t+o start face detection
    if st.button('Start Face Detection'):
        detect_faces(selected_color, min_neighbors, scale_factor)

if __name__ == "__main__":
    main()
