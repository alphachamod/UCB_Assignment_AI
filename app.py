import cv2
import streamlit as st
import os
from datetime import datetime
from keras_facenet import FaceNet
import numpy as np
import uuid
from streamlit_option_menu import option_menu
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv

import modifyPerson
from activityHistory import display_entry_logs
from blacklist import show_blacklisted_persons
from personLogs import list_saved_persons

logo_path = "assests/logo2.png"
title_path = "assests/title.png"

with st.sidebar:
    st.image(logo_path, width=200)
    st.image(title_path, use_column_width=True)

    selected = option_menu(
        menu_title="Main Menu",
        options=["Main Feed", "Update", "Blacklist Section", "Entry History", "Person Logs"],
        icons=["camera-video", "pencil-square", "ban", "clock-history", "person-lines-fill"],
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


# Create a directory to store CSV files
CSV_DIR = "entry_logs"
os.makedirs(CSV_DIR, exist_ok=True)


def record_entry(name, entry_type, entry_time):
    try:
        # Create a CSV file with today's date as filename
        today_date = datetime.now().strftime("%Y-%m-%d")
        csv_filename = os.path.join(CSV_DIR, f"{today_date}.csv")

        # Check if the CSV file already exists
        file_exists = os.path.isfile(csv_filename)

        # Check if the name already exists in the CSV file
        name_exists = False
        if file_exists:
            df = pd.read_csv(csv_filename)
            if name in df['Name'].values:
                name_exists = True

        # Check if the entry is late or on time for staff
        status = ""
        minutes_late = 0
        if entry_type == "Staff":
            if entry_time <= "09:00:00":
                status = "On Time"
            else:
                status = "Late"
                # Calculate minutes late
                entry_time_obj = datetime.strptime(entry_time, "%H:%M:%S")
                late_time_obj = datetime.strptime("09:00:00", "%H:%M:%S")
                minutes_late = (entry_time_obj - late_time_obj).total_seconds() / 60

        # Write entry to CSV file if name doesn't exist
        if not name_exists:
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)

                # Write header if the file is newly created
                if not file_exists:
                    writer.writerow(['Name', 'Type', 'Entry Time', 'Exit Time', 'Duration (hours)', 'Status', 'Minutes Late'])

                # Write entry for the current person
                writer.writerow([name, entry_type, entry_time, '', '', status, minutes_late])

                # Display message in the sidebar based on the entry type and status
                if entry_type == "Staff":
                    if status == "Late":
                        st.sidebar.write(f"{name} is {int(minutes_late)} minutes late")
                    else:
                        st.sidebar.write(f"{name} is on time")
                else:
                    st.sidebar.write("Entry time saved for:", name)
        else:
            print("Name already exists in the CSV file:", name)

    except Exception as e:
        print(f"Error occurred while creating the CSV file: {e}")


def update_exit_time(name, exit_time):
    try:
        # Create a CSV file with today's date as filename
        today_date = datetime.now().strftime("%Y-%m-%d")
        csv_filename = os.path.join(CSV_DIR, f"{today_date}.csv")

        # Read the CSV file
        df = pd.read_csv(csv_filename)

        # Find the row corresponding to the person's name
        row_index = df.index[df['Name'] == name].tolist()

        # If the person's name is found in the CSV file
        if row_index:
            entry_time = datetime.strptime(df.at[row_index[0], 'Entry Time'], "%H:%M:%S")

            # Check if the exit time is already recorded
            if pd.isnull(df.at[row_index[0], 'Exit Time']):
                # Calculate the time difference
                time_difference = (datetime.now() - entry_time).seconds

                # Check if the time difference is greater than 10 seconds
                if time_difference > 10:
                    df.at[row_index[0], 'Exit Time'] = exit_time
                    df.at[row_index[0], 'Duration (hours)'] = calculate_duration(df.at[row_index[0], 'Entry Time'], exit_time)
                    st.sidebar.write("Exit time updated successfully for:", name)
                    # Write updated entries back to CSV file
                    df.to_csv(csv_filename, index=False)
            else:
                print("Exit time already recorded for:", name)
        else:
            print("Name not found in CSV file:", name)

    except Exception as e:
        print(f"Error occurred while updating exit time: {e}")

def calculate_duration(entry_time, exit_time):
    entry_datetime = datetime.strptime(entry_time, "%H:%M:%S")
    exit_datetime = datetime.strptime(exit_time, "%H:%M:%S")
    duration = exit_datetime - entry_datetime
    duration_hours = duration.total_seconds() / 3600
    return duration_hours


if selected == "Main Feed":
    # Define the directory where embedding files are saved
    SAVE_DIR = "captured_data"

    # Define the size of the embeddings
    embedding_size = 128  # For example, if the embeddings are 128-dimensional

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

    # Load saved embedding files
    saved_embedding_files = []
    for root, dirs, files in os.walk(SAVE_DIR):
        for file in files:
            if file.endswith("_embeddings.npy"):
                saved_embedding_files.append(os.path.join(root, file))

    # Initialize saved_embeddings as None
    saved_embeddings = None

    # Load saved embeddings if files are found
    if saved_embedding_files:
        saved_embeddings_list = []
        for saved_embedding_file in saved_embedding_files:
            embeddings = np.load(saved_embedding_file)
            saved_embeddings_list.append(embeddings)
        saved_embeddings = np.concatenate(saved_embeddings_list, axis=0)

    # Check if saved_embeddings is None or empty
    if saved_embeddings is None or saved_embeddings.size == 0:
        print("No saved embeddings found.")
    else:
        print("Saved embeddings loaded successfully.")

    frame_count = 0
    MAX_FRAMES = 50

    # Input field for name
    name_input = st.text_input("Enter the name:")
    name_type = st.radio("Type:", ["Staff", "Visitor", "Special"])
    print("Selected type:", name_type)

    # Place the button in the main panel
    capture_button = st.button("Capture 50 Faces")

    embedder = FaceNet()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    entry_recorded = False  # Flag to check if entry is recorded

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
                    record_type = embedding_filename.split("_")[1]
                    break

            # Update the entry_recorded variable when a new entry is recorded
            if recognized:
                name = recognized_name
                entry_type = record_type
                entry_time = datetime.now().strftime("%H:%M:%S")
                record_entry(name, entry_type, entry_time)

                # Calculate exit time
                exit_time = datetime.now().strftime("%H:%M:%S")

                # Update exit time if time difference is greater than 10 seconds
                update_exit_time(name, exit_time)

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
            name = name_input.strip().replace(" ",
                                              "_")  # Remove leading/trailing spaces and replace spaces with underscores
            type = name_type.strip().replace(" ",
                                             "_")  # Remove leading/trailing spaces and replace spaces with underscores
            # Generate a unique ID for this capture session
            timestamp = datetime.now().strftime("%Y-%m-%d")
            unique_id = uuid.uuid4().hex[:8]

            # Include "notblacklisted" in the folder name
            unique_folder_name = f"{name}_{type}_notblacklisted_{timestamp}_{unique_id}"

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
            embeddings_file_path = os.path.join(session_dir, f"{name}_{type}_notblacklisted_embeddings.npy")
            np.save(embeddings_file_path, all_embeddings)


            # Show notification in the sidebar
            show_notification(f"{name} ({type}) has been added")
            capture_button = False

if selected == "Update":
    # Define the directory where files are saved
    SAVE_DIR = "captured_data"

    # List all files in the directory
    all_files = os.listdir(SAVE_DIR)

    # Select a file to edit or delete
    selected_file = st.selectbox("Select File to Edit/Delete", all_files)

    # Input fields to enter new name and type
    new_name = st.text_input("Enter New Name", value=selected_file.split("_")[0])
    new_type = st.selectbox("Select New Type", options=["Staff", "Visitor", "Special"],
                            index=["Staff", "Visitor", "Special"].index(selected_file.split("_")[1]))

    # Checkbox to select whether the entry is blacklisted or not
    blacklisted = st.checkbox("Blacklisted")

    # Button to save changes
    if st.button("Save Changes"):
        # Call the function to edit and save file details
        modifyPerson.edit_entry_details(SAVE_DIR, selected_file, new_name, new_type, blacklisted)

    # Checkbox to select whether to delete the selected person and folder
    delete_checkbox = st.checkbox("Delete Person and Folder")

    # Button to confirm deletion
    if delete_checkbox:
        if st.button("Confirm Deletion"):
            # Call the function to delete the selected person and folder
            modifyPerson.delete_person_and_folder(SAVE_DIR, selected_file)

if selected == "Blacklist Section":
    st.title("Blacklist Section")
    SAVE_DIR = "captured_data"
    ENTRY_LOGS_DIR = "entry_logs"
    show_blacklisted_persons(SAVE_DIR, ENTRY_LOGS_DIR)

if selected == "Entry History":
    st.title("Entry Logs Viewer")

    # Date input widget to select the date
    selected_date = st.date_input("Select Date", datetime.now())

    # Call the function to display entry logs for the selected date
    display_entry_logs(selected_date)


if selected == "Person Logs":

    st.title("List of Saved Persons")

    # Define the directories where files and entry logs are saved
    SAVE_DIR = "captured_data"
    ENTRY_LOGS_DIR = "entry_logs"

    # Input field to filter by type
    filter_type = st.selectbox("Filter by Type", options=["All", "Staff", "Visitor", "Special"])

    # List all saved persons in a table
    if filter_type == "All":
        st.write(list_saved_persons(SAVE_DIR, ENTRY_LOGS_DIR))
    else:
        st.write(list_saved_persons(SAVE_DIR, ENTRY_LOGS_DIR, filter_type))


