import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from saliency.static_salience import salience_static


def show_page_1():

    st.header("Cartes de saillance statiques")

    beta = st.slider(
        "Sensibilité β (softmax)",
        0.001,
        0.2,
        0.02,
        0.001
    )

    def aoi_to_matrix(values):
        return np.array(values).reshape(10,10)

    games = {
        1: "BCG",
        2: "BCG +",
        3: "BCG -"
    }

    cols = st.columns(3)

    for i, game in enumerate(games):

        result = salience_static(game, beta=beta)

        matrix = aoi_to_matrix(result["q"])

        fig, ax = plt.subplots(figsize=(5,5))

        sns.heatmap(
            matrix,
            cmap="viridis",
            square=True,
            cbar=True,
            ax=ax
        )

        ax.set_title(games[game])

        cols[i].pyplot(fig)