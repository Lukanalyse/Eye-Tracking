import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st

from src.stage_1.page_1 import show_stage1_page

st.set_page_config(page_title="Stage 1 - Eye Tracking BCG", layout="wide")

st.title("Stage 1 - BR computationnelle + blur spatial")
st.markdown("Analyse du Stage 1 avec diffusion spatiale gaussienne.")

show_stage1_page()