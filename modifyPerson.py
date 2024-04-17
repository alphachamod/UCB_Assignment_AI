import os
import shutil
import streamlit as st
import re

# Function to edit and save details for pre-captured entries
def edit_entry_details(SAVE_DIR, selected_file, new_name, new_type, blacklisted):
    try:
        # Extract components from the selected file name
        match = re.match(r'(.+?)_(.+?)_(blacklisted|notblacklisted)_(\d{4}-\d{2}-\d{2})_(\w+)', selected_file)
        if match:
            old_name, old_type, old_blacklist_status, old_timestamp, old_unique_id = match.groups()
            old_folder_name = f"{old_name}_{old_type}_{old_blacklist_status}_{old_timestamp}_{old_unique_id}"
            old_folder_path = os.path.join(SAVE_DIR, old_folder_name)

            # Construct the new folder name with updated name, type, blacklisted status, and old unique id
            blacklisted_suffix = "blacklisted" if blacklisted else "notblacklisted"
            new_folder_name = f"{new_name}_{new_type}_{blacklisted_suffix}_{old_timestamp}_{old_unique_id}"

            # Construct the new folder path
            new_folder_path = os.path.join(SAVE_DIR, new_folder_name)

            # Rename the folder
            os.rename(old_folder_path, new_folder_path)

            st.success("Entry details updated successfully.")
        else:
            st.error("Invalid file name format.")
    except Exception as e:
        st.error(f"Error occurred while updating entry details: {e}")

# Function to delete the selected person and their folder
def delete_person_and_folder(SAVE_DIR, selected_file):
    try:
        # Construct the folder path
        folder_path = os.path.join(SAVE_DIR, os.path.splitext(selected_file)[0])

        # Remove the folder and its contents
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        st.success("Person and folder deleted successfully.")
    except Exception as e:
        st.error(f"Error occurred while deleting person and folder: {e}")