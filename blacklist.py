import os
import pandas as pd
import streamlit as st
from PIL import Image

def show_blacklisted_persons(SAVE_DIR, ENTRY_LOGS_DIR):
    # Initialize an empty list to store blacklisted persons
    blacklisted_persons = []

    # Iterate over each file in the save directory
    for file_name in os.listdir(SAVE_DIR):
        try:
            # Extract name, type, date, blacklisted status, and ID from the file name
            parts = os.path.splitext(file_name)[0].split('_')
            name = parts[0].capitalize()
            file_type = parts[1]
            date = parts[3]
            blacklisted = parts[2]
            unique_id = parts[4].upper()
        except ValueError:
            # Handle cases where the file name structure is different
            continue

        # Check if the person is blacklisted
        if blacklisted == "blacklisted":
            # Search for face images in the person's folder
            person_folder = os.path.join(SAVE_DIR, file_name)
            face_images = [f for f in os.listdir(person_folder) if f.startswith("face_")]
            black_status = True

            # Get the path of the first found face image
            face_image_path = os.path.join(person_folder, face_images[0]) if face_images else None

            # Check if the image path exists and load the image
            if face_image_path and os.path.isfile(face_image_path):
                # Load the image using Pillow
                img = Image.open(face_image_path)
                # Resize the image
                img = img.resize((150, 150))

                blacklisted_persons.append(
                    {'ID': unique_id, 'Name': name, 'Type': file_type, 'Date Added': date, 'Face Image': img})
            else:
                st.warning(f"Face image not found for {name}.")

    # Convert the list of dictionaries to a DataFrame
    df_blacklisted = pd.DataFrame(blacklisted_persons)

    # Display the blacklisted persons in two columns
    col1, col2 = st.columns(2)
    for index, row in df_blacklisted.iterrows():
        # Display person details in the first column
        col1.write(f"**ID:** {row['ID']}")
        col1.write(f"**Name:** {row['Name']}")
        col1.write(f"**Type:** {row['Type']}")
        col1.write(f"**Date Added:** {row['Date Added']}")
        # Display the image in the second column
        col2.write(" ")
        col2.image(row['Face Image'], caption='Face', width=150)
        col1.divider()


