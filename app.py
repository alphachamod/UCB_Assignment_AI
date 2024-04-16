import cv2
import streamlit as st
import os
from datetime import datetime
from keras_facenet import FaceNet
import numpy as np
import uuid
from streamlit_option_menu import option_menu
from sklearn.metrics.pairwise import cosine_similarity

logo_path = "logo2.png"
title_path = "title.png"

with st.sidebar:
    st.image(logo_path, width=300)
    st.image(title_path, use_column_width=True)

    selected = option_menu(
        menu_title="Main Menu",
        options=["Main Feed", "Add New", "Entry History"],
        icons=["camera-video", "person-add", "clock-history"],
        menu_icon="house",
        default_index=0,
        styles={"H1": {"color": "orange", "font-size": "25px"}},
    )

def show_notification(message, success=True):
    notification_box = st.empty()
    if success:
        notification_box.success(f":white_check_mark: {message}")
    else:
        notification_box.error(message)


if selected == "Main Feed":
    st.title("Live Camera Feed")
    FRAME_WINDOW = st.image([])

    # Use Streamlit caching to initialize camera once
    @st.cache(allow_output_mutation=True)
    def initialize_camera():
        return cv2.VideoCapture(0)

    camera = initialize_camera()

    # Directory to save captured data
    SAVE_DIR = "captured_data"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load saved embeddings and filenames
    saved_embedding_files = []
    saved_embeddings = []

    for root, dirs, files in os.walk(SAVE_DIR):
        for file in files:
            if file.endswith("_embeddings.npy"):
                saved_embedding_files.append(os.path.join(root, file))
                embeddings = np.load(os.path.join(root, file))
                saved_embeddings.append(embeddings)

    saved_embeddings = np.concatenate(saved_embeddings)

    frame_count = 0
    MAX_FRAMES = 50

    # Input field for name
    name_input = st.text_input("Enter the name:")
    name_type = st.selectbox("Type:", ["Staff", "Visitor", "Special"])

    # Place the button in the main panel
    capture_button = st.button("Capture 50 Faces")

    embedder = FaceNet()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = camera.read()
        if not ret:
            st.write('Camera disconnected!')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            embedding = embedder.embeddings([face_img])[0]

            # Compare face embeddings with saved embeddings
            similarities = cosine_similarity([embedding], saved_embeddings)

            # Inside the loop where we compare embeddings and display names
            recognized = False
            max_similarity = 0.0
            recognized_name = ""
            for saved_embedding_file in saved_embedding_files:
                saved_embeddings = np.load(saved_embedding_file)
                similarities = cosine_similarity([embedding], saved_embeddings)
                if np.max(similarities) > 0.7:
                    recognized = True
                    max_similarity = np.max(similarities)
                    embedding_filename = os.path.basename(saved_embedding_file)
                    recognized_name = embedding_filename.split("_")[0]
                    break

            # Draw rectangle around the face
            if recognized:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle for recognized faces
                cv2.putText(frame, f"{recognized_name} (Similarity: {max_similarity:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle for unrecognized faces
                cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        FRAME_WINDOW.image(frame)

        if capture_button and name_input:
            name = name_input.strip().replace(" ", "_")  # Remove leading/trailing spaces and replace spaces with underscores
            type = name_type.strip().replace(" ", "_")  # Remove leading/trailing spaces and replace spaces with underscores
            # Generate a unique ID for this capture session
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            unique_id = uuid.uuid4().hex[:8]
            unique_folder_name = f"{name}_{type}_{timestamp}_{unique_id}"
            session_dir = os.path.join(SAVE_DIR, unique_folder_name)
            os.makedirs(session_dir, exist_ok=True)

            all_embeddings = []
            while frame_count < MAX_FRAMES:
                _, frame = camera.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Face detection using Haar Cascade
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face_img = frame[y:y + h, x:x + w]
                    embedding = embedder.embeddings([face_img])[0]

                    # Save the image of the face
                    cv2.imwrite(os.path.join(session_dir, f"face_{frame_count}.jpg"), face_img)

                    all_embeddings.append(embedding)

                    frame_count += 1

            # Convert the list of embeddings to a numpy array
            all_embeddings = np.array(all_embeddings)

            # Save all embeddings to a single file inside the generated folder
            embeddings_file_path = os.path.join(session_dir, f"{name}_{type}_embeddings.npy")
            np.save(embeddings_file_path, all_embeddings)

            # Show notification in the sidebar
            show_notification(f"{name} ({type}) has been added")
            capture_button = False

if selected == "Add New":
    st.title(f"You have selected {selected}")
if selected == "Entry History":
    st.title(f"You have selected {selected}")
