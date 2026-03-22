import streamlit as st
import sqlite3
import os
import pandas as pd
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data.db")

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(), pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df_faces = pd.read_sql_query(
                "SELECT id, first_seen, last_seen FROM faces ORDER BY last_seen DESC", conn)
            df_events = pd.read_sql_query(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT 500", conn)
        return df_faces, df_events
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def resolve_image(path):
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return path if os.path.exists(path) else None

def load_thumb(path, size=180):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
        img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception:
        return None

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {
    background-color: #F8F9FB !important;
    font-family: 'Inter', sans-serif !important;
    color: #1E293B !important;
}
#MainMenu, footer, header { visibility: hidden; }
.ph { text-align:center; padding:28px 0 0 0; }
.ph h1 { font-size:1.75rem; font-weight:700; color:#1E293B; margin:0; }
.ph p  { font-size:0.88rem; color:#64748B; margin:6px 0 0 0; }
.hr   { border:none; border-top:1px solid #E2E8F0; margin:16px 0; }
.sh   { font-size:1rem; font-weight:600; color:#1E293B;
        border-left:3px solid #2563EB; padding-left:10px; margin:0 0 12px 0; }
.sc   { background:#FFF; border:1px solid #E2E8F0; border-radius:10px;
        padding:16px 20px; text-align:center;
        box-shadow:0 1px 3px rgba(0,0,0,.05); margin-bottom:10px; }
.sc-n { font-size:1.9rem; font-weight:700; color:#2563EB; }
.sc-l { font-size:0.7rem; color:#94A3B8; text-transform:uppercase;
        letter-spacing:.8px; margin-top:3px; }
.fc   { background:#FFF; border:1px solid #E2E8F0; border-radius:12px;
        padding:12px 10px 10px 10px; text-align:center;
        box-shadow:0 1px 3px rgba(0,0,0,.05); margin-bottom:14px;
        transition:box-shadow .2s, transform .2s; }
.fc:hover { box-shadow:0 6px 20px rgba(37,99,235,.14); transform:translateY(-3px); }
.fc-id   { font-size:0.72rem; font-weight:700; color:#2563EB;
           font-family:monospace; word-break:break-all; margin:8px 0 4px 0; }
.fc-meta { font-size:0.67rem; color:#64748B; line-height:1.7; text-align:left; }
.stTabs [data-baseweb="tab-list"] {
    background:#FFF; border:1px solid #E2E8F0;
    border-radius:10px; padding:4px; gap:2px;
}
.stTabs [data-baseweb="tab"]          { border-radius:7px; color:#64748B; font-size:.87rem; font-weight:500; }
.stTabs [aria-selected="true"]        { background:#2563EB !important; color:#FFF !important; }
section[data-testid="stSidebar"]      { background:#FFF !important; border-right:1px solid #E2E8F0; }
section[data-testid="stSidebar"] *    { color:#1E293B !important; }
.stDataFrame { border-radius:10px; overflow:hidden; border:1px solid #E2E8F0; }
</style>
"""

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="AI Face Tracking System", page_icon="🎯",
                       layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="ph">
        <h1>🎯 AI Face Tracking &amp; Visitor Analytics</h1>
        <p>Real-time Face Detection · Tracking · Visitor Counting</p>
    </div><div class="hr"></div>""", unsafe_allow_html=True)

    if not os.path.exists(DB_PATH):
        st.error("Database not found. Run `python app.py` first.")
        return

    df_faces, df_events = load_data()

    # ── Stats row ──────────────────────────────────────────────────────────────
    total_faces = len(df_faces)
    entry_count = int((df_events["event_type"] == "entry").sum()) if not df_events.empty else 0
    exit_count  = int((df_events["event_type"] == "exit").sum())  if not df_events.empty else 0

    main_col, stat_col = st.columns([7, 3], gap="large")

    with main_col:
        st.markdown('<div class="sh">📋 Event Log</div>', unsafe_allow_html=True)
        if df_events.empty:
            st.info("No events yet.")
        else:
            show = [c for c in ["timestamp", "face_id", "event_type", "image_path"]
                    if c in df_events.columns]
            st.dataframe(df_events[show], use_container_width=True,
                         height=300, hide_index=True)

    with stat_col:
        st.markdown('<div class="sh">📊 Statistics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sc"><div class="sc-n">{total_faces}</div><div class="sc-l">Unique Visitors</div></div>
        <div class="sc"><div class="sc-n">{entry_count}</div><div class="sc-l">Entry Events</div></div>
        <div class="sc"><div class="sc-n">{exit_count}</div><div class="sc-l">Exit Events</div></div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

    # ── Registered Faces ───────────────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sh">👤 Registered Faces</div>', unsafe_allow_html=True)

    if df_faces.empty:
        st.info("No faces registered yet.")
        return

    # Build image path lookup from entry events
    img_lookup = {}
    if not df_events.empty:
        for fid, grp in df_events[df_events["event_type"] == "entry"].groupby("face_id"):
            img_lookup[fid] = grp.iloc[-1]["image_path"]

    # ── Build list of faces that have a valid image — sorted newest first ──────
    valid_faces = []
    for _, row in df_faces.iterrows():          # df already ORDER BY last_seen DESC
        face_id  = row["id"]
        img_path = resolve_image(img_lookup.get(face_id))
        if not img_path:                        # skip faces with no saved image
            continue
        valid_faces.append({
            "id":         face_id,
            "first_seen": str(row.get("first_seen", "—"))[:16],
            "last_seen":  str(row.get("last_seen",  "—"))[:16],
            "img_path":   img_path,
            "short_id":   face_id.split("_")[-1] if "_" in face_id else face_id,
        })

    if not valid_faces:
        st.info("No face images saved yet.")
        return

    # ── Render grid: cols[i % 4] — no empty cards, no stacking ───────────────
    COLS = 4
    cols = st.columns(COLS, gap="medium")

    for i, face in enumerate(valid_faces):
        with cols[i % COLS]:
            st.markdown('<div class="fc">', unsafe_allow_html=True)

            thumb = load_thumb(face["img_path"], size=180)
            if thumb:
                st.image(thumb, use_container_width=True)

            st.markdown(f"""
            <div class="fc-id">#{face['short_id']}</div>
            <div class="fc-meta">
                🕐 <b>First:</b> {face['first_seen']}<br>
                🕔 <b>Last:</b>&nbsp; {face['last_seen']}
            </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
