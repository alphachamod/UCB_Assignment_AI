import os
import pandas as pd
import streamlit as st


# Function to list all saved persons in a table
def list_saved_persons(SAVE_DIR, ENTRY_LOGS_DIR, filter_type=None):
    # Initialize an empty list to store person details
    person_details = []

    # Iterate over each file in the save directory
    for file_name in os.listdir(SAVE_DIR):
        try:
            # Extract name, type, date, and blacklisted status from the file name
            parts = os.path.splitext(file_name)[0].split('_')
            name = parts[0].capitalize()
            file_type = parts[1]
            blacklisted = parts[2]
            date = parts[3]
            uid = parts[4].upper()

        except ValueError:
            # Handle cases where the file name structure is different
            name = os.path.splitext(file_name)[0]
            file_type = "Unknown"
            date = "Unknown"
            blacklisted = "No"

        if blacklisted == "notblacklisted":
            blacklisted_stat = " No"
        else:
            blacklisted_stat = " Yes"

        # Check if filter_type is None or matches the file_type
        if filter_type is None or filter_type == file_type:
            # Append person details to the list
            person_details.append({'ID': uid,
                                   'Name': name,
                                   'Type': file_type,
                                   'Date Added': date,
                                   'Blacklisted': blacklisted_stat})

     # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(person_details)

    return df
