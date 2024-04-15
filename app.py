import cv2
import mediapipe as mp
import streamlit as st
import os
import pickle
from datetime import datetime
import uuid
from streamlit_option_menu import option_menu

logo_path = "logo2.png"
title_path = "title.png"

with st.sidebar:

    st.image(logo_path, width=300)
    st.image(title_path, use_column_width=True)

    # st.title("Face Recognition System")

    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Main Feed", "Add New", "Entry History"],  # required
        icons=["camera-video", "person-add", "clock-history"],  # optional
        menu_icon="house",  # optional
        default_index=0,  # optional
        styles={
            "H1": {"color": "orange", "font-size": "25px"},
        },
    )

if selected == "Main Feed":
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    st.title("Live Camera Feed")
    FRAME_WINDOW = st.image([])

    # Use Streamlit caching to initialize camera once
    @st.cache(allow_output_mutation=True)
    def initialize_camera():
        return cv2.VideoCapture(0)


    camera = initialize_camera()

    # Directory to save captured frames and embeddings
    SAVE_DIR = "captured_data"
    os.makedirs(SAVE_DIR, exist_ok=True)

    frame_count = 0
    MAX_FRAMES = 50
    face_landmarks_list = []

    # Input field and dropdown for name
    name_input = st.text_input("Enter the name:")
    name_type = st.selectbox("Type:", ["Staff", "Visitor", "Special"])

    # Place the button in the main panel
    capture_button = st.button("Capture 50 Faces")

    # Custom notification box
    notification_box = st.empty()


    def show_notification(message, success=True):
        if success:
            notification_box.success(f":white_check_mark: {message}")
        else:
            notification_box.error(message)


    def capture_faces(name):
        global frame_count
        global face_landmarks_list

        # Generate a unique ID for this capture session
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_id = uuid.uuid4().hex[:8]
        unique_folder_name = f"{timestamp}_{unique_id}"

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            while frame_count < MAX_FRAMES:
                _, frame = camera.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Use MediaPipe for face detection
                with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                    results = face_detection.process(frame_rgb)

                    if results.detections:
                        for detection in results.detections:
                            ih, iw, _ = frame.shape
                            bboxC = detection.location_data.relative_bounding_box
                            xmin, ymin, width, height = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                int(bboxC.width * iw), int(bboxC.height * ih)

                            # Draw rectangle around the face
                            cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)

                            # Crop the frame to capture only the face area
                            face_frame = frame[ymin:ymin + height, xmin:xmin + width]

                            # Convert frame to RGB for face mesh
                            frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

                            # Extract facial features using MediaPipe FaceMesh
                            results_mesh = face_mesh.process(frame_rgb)

                            if results_mesh.multi_face_landmarks:
                                # Flatten landmarks
                                landmarks = [list((lm.x, lm.y, lm.z)) for lm in
                                             results_mesh.multi_face_landmarks[0].landmark]
                                face_landmarks_list.append(landmarks)

                            # Save cropped face frame locally
                            save_path = os.path.join(SAVE_DIR, f"{name}_{unique_folder_name}")
                            os.makedirs(save_path, exist_ok=True)
                            frame_filename = os.path.join(save_path, f"face_{frame_count}.jpg")
                            cv2.imwrite(frame_filename, cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR))
                            frame_count += 1

                FRAME_WINDOW.image(frame)

        # Save face landmarks as a pickle file inside the subfolder
        pickle_file_path = os.path.join(save_path, f"{name}_{unique_folder_name}_landmarks.pkl")
        with open(pickle_file_path, "wb") as f:
            pickle.dump(face_landmarks_list, f)

        # Show notification in the sidebar
        show_notification(f"{name} has been added")


    # Streaming loop
    while True:
        ret, frame = camera.read()
        if not ret:
            st.write('Camera disconnected!')
            break

        # Use MediaPipe for face detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(frame)

            if results.detections:
                for detection in results.detections:
                    ih, iw, _ = frame.shape
                    bboxC = detection.location_data.relative_bounding_box
                    xmin, ymin, width, height = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)

                    # Draw rectangle around the face
                    cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)

                    # Add text "Unknown" above the rectangle if the person is unknown
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'Unknown', (xmin, ymin - 10), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame)

        if capture_button and name_input:
            name = f"{name_input}_{name_type}"
            capture_faces(name)
            capture_button = False

if selected == "Add New":
    st.title(f"You have selected {selected}")
if selected == "Entry History":
    st.title(f"You have selected {selected}")
