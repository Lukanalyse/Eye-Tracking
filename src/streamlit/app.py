import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

# import pages
from pages.page_1 import show_page_1

st.set_page_config(
    page_title="Eye Tracking BCG",
    layout="wide"
)

st.title("Modélisation cognitive du Beauty Contest Game")
st.markdown("Analyse des données de eye-tracking et des cartes de saillance théoriques.")


# =========================
# Navigation
# =========================

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Carte de saillance théorique",
    ]
)


# =========================
# Pages
# =========================

if page == "Carte de saillance théorique":
    show_page_1()