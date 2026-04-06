import sys
from pathlib import Path

# Add project root so `from src...` imports work when launching Streamlit from this file.
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st

# import pages
from src.pages.page_1 import show_page_1
from src.pages.page_4 import show_page_4
from src.pages.page_5 import show_page_5

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
        "Modelisation dynamique (Stage 1)",
"Modelisation dynamique (Stage 2 - Memoire locale)",
    ]
)


# =========================
# Pages
# =========================

if page == "Carte de saillance théorique":
    show_page_1()
elif page == "Modelisation dynamique (Stage 1)":
    show_page_4()
elif page == "Modelisation dynamique (Stage 2 - Memoire locale)":
    show_page_5()