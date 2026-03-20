import streamlit as st
import sqlite3
import os
import pandas as pd
from datetime import datetime
from PIL import Image

# Get the absolute path to the database
# Directory of this script: d:\ai_face_tracking_system\ui
# Database path: d:\ai_face_tracking_system\data.db
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(), pd.DataFrame()
        
    with get_connection() as conn:
        try:
            df_faces = pd.read_sql_query("SELECT id, first_seen, last_seen FROM faces", conn)
            df_events = pd.read_sql_query("SELECT * FROM events ORDER BY timestamp DESC", conn)
            return df_faces, df_events
        except Exception as e:
            st.error(f"Error reading database: {e}")
            return pd.DataFrame(), pd.DataFrame()

def main():
    st.set_page_config(page_title="Face Database Viewer", layout="wide")
    st.title("👤 Face Database Viewer")
    
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at {DB_PATH}. Please run the system first.")
        # Debug: List files in BASE_DIR
        st.write(f"Current Base Directory: {BASE_DIR}")
        st.write("Files in directory:", os.listdir(BASE_DIR))
        return

    df_faces, df_events = load_data()

    st.sidebar.header("Statistics")
    st.sidebar.metric("Total Unique Faces", len(df_faces))
    st.sidebar.metric("Total Events Logged", len(df_events))

    tabs = st.tabs(["Registered Faces", "Recent Events"])

    with tabs[0]:
        st.header("Registered Faces")
        if df_faces.empty:
            st.info("No faces registered yet.")
        else:
            cols = st.columns(4)
            for idx, row in df_faces.iterrows():
                with cols[idx % 4]:
                    face_id = row['id']
                    st.subheader(f"ID: {face_id}")
                    
                    # Try to find the first entry image for this face
                    face_image_path = None
                    face_events = df_events[df_events['face_id'] == face_id]
                    if not face_events.empty:
                        entry_events = face_events[face_events['event_type'] == 'entry']
                        if not entry_events.empty:
                            face_image_path = entry_events.iloc[-1]['image_path']
                    
                    # Ensure path is absolute if it was stored as relative
                    if face_image_path and not os.path.isabs(face_image_path):
                        face_image_path = os.path.join(BASE_DIR, face_image_path)

                    if face_image_path and os.path.exists(face_image_path):
                        img = Image.open(face_image_path)
                        st.image(img, use_container_width=True)
                    else:
                        st.warning(f"No image at {face_image_path}")
                    
                    st.write(f"**First Seen:** {row['first_seen']}")
                    st.write(f"**Last Seen:** {row['last_seen']}")
                    st.divider()

    with tabs[1]:
        st.header("Recent Events")
        if df_events.empty:
            st.info("No events logged yet.")
        else:
            st.dataframe(df_events, use_container_width=True)

if __name__ == "__main__":
    main()
