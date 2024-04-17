import os
import streamlit as st
import pandas as pd


CSV_DIR = "entry_logs"
os.makedirs(CSV_DIR, exist_ok=True)
def display_entry_logs(selected_date):
    try:
        # Convert the selected date to string format
        selected_date_str = selected_date.strftime("%Y-%m-%d")

        # Create the CSV filename based on the selected date
        csv_filename = os.path.join(CSV_DIR, f"{selected_date_str}.csv")

        # Check if the CSV file exists for the selected date
        if os.path.isfile(csv_filename):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_filename)

            # Display the DataFrame in a table format
            st.write("Entry Logs for", selected_date_str)
            st.write(df)
        else:
            st.write("No entry logs found for", selected_date_str)
    except Exception as e:
        st.error(f"Error occurred while retrieving entry logs: {e}")