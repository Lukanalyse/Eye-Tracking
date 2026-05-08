# src/pages/page_0.py
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from src.data_processing.data_processing import get_data_dirs
from src.pages.page_1 import (
    AOI_COUNT,
    GRID_SIZE,
    aoi_to_coordinates,
    aoi_to_matrix,
    discover_player_files,
    extract_game_number,
    is_supported_game_sheet,
    load_workbook_sheets,
)
from src.stage_0.stage_0 import (
    Stage0Params,
    stage_0_step,
)
from src.stage_0.score_0 import (
    Score0Params,
    stage0_br_count_distribution,
)
from src.saliency.static_salience import get_game_metadata


def _minmax_for_display(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    return (values - v_min) / (v_max - v_min + 1e-12)


def _plot_heatmap(
    values: np.ndarray,
    title: str,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    current_point: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> plt.Figure:
    matrix = aoi_to_matrix(_minmax_for_display(values))

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    sns.heatmap(matrix, cmap=cmap, square=True, cbar=True, ax=ax)

    if x_coords is not None and y_coords is not None and len(x_coords) > 0:
        ax.plot(x_coords, y_coords, color="white", linewidth=1.2, alpha=0.95)

    if current_point is not None:
        ax.scatter(
            current_point[0],
            current_point[1],
            color="red",
            s=70,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, 0)
    ax.set_xticks(np.arange(GRID_SIZE) + 0.5)
    ax.set_xticklabels(np.arange(1, GRID_SIZE + 1))
    ax.set_yticks(np.arange(GRID_SIZE) + 0.5)
    ax.set_yticklabels(np.arange(0, GRID_SIZE))
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_title(title)

    return fig


def _init_page0_state(selection_key: str) -> None:
    if st.session_state.get("page0_selection_key") != selection_key:
        st.session_state.page0_selection_key = selection_key
        st.session_state.page0_frame_idx = 0
        st.session_state.page0_is_playing = False
        st.session_state.page0_last_rerun_time = None
        st.session_state.page0_play_progress = 0.0

    st.session_state.setdefault("page0_frame_idx", 0)
    st.session_state.setdefault("page0_is_playing", False)
    st.session_state.setdefault("page0_last_rerun_time", None)
    st.session_state.setdefault("page0_play_progress", 0.0)


def show_page_0() -> None:
    st.header("Stage 0 - Salience statique + BR computationnelle")
    st.caption(
        "Modèle minimal : S_t(y) = λ0 S_stat(y) + λ1 BR_comp_t(y), avec λ0 + λ1 = 1. "
        "Le score BR est maintenant calculé à partir de la distance entre la transition observée "
        "et la cible stratégique T_g(x_t)."
    )

    project_root = Path(__file__).resolve().parents[2]
    data_dir, output_dir = get_data_dirs()
    player_files = discover_player_files(str(project_root))

    if not player_files:
        st.warning("Aucun fichier Excel joueur trouvé.")
        st.info(
            f"Dossiers scannés :\n- {project_root}\n- {output_dir}\n- {data_dir}\n\n"
            "Place les fichiers Excel joueurs dans l'un de ces dossiers."
        )
        return

    left_col, center_col = st.columns([1.15, 2.25])

    with left_col:
        st.subheader("Sélection")
        player_name = st.selectbox("Joueur", options=list(player_files.keys()), key="page0_player")
        workbook_path = player_files[player_name]

        workbook = load_workbook_sheets(workbook_path)
        sheet_names = [name for name in workbook if is_supported_game_sheet(name)]

        if not sheet_names:
            st.warning("Aucune feuille GAME1..GAME6 exploitable dans ce fichier.")
            return

        sheet_name = st.selectbox(
            "Jeu (sheet)",
            options=sorted(sheet_names, key=extract_game_number),
            key="page0_sheet",
        )

        st.markdown("**Paramètres Stage 0**")
        lambda_stat = st.slider("lambda_stat", 0.0, 1.0, 0.5, 0.01, key="page0_lambda_stat")
        lambda_br = st.slider("lambda_br", 0.0, 1.0, 0.5, 0.01, key="page0_lambda_br")
        sigma_comp = st.slider("sigma_comp", 0.2, 25.0, 8.0, 0.1, key="page0_sigma_comp")

        st.markdown("**Paramètres Score 0**")
        sigma_score = st.slider("sigma_score", 0.5, 10.0, 2.0, 0.1, key="page0_sigma_score")

        st.markdown("**Lecture dynamique**")
        speed = st.slider("Vitesse de lecture", 0.05, 5.0, 0.5, 0.05, key="page0_speed")

    fixations = workbook[sheet_name]
    if fixations.empty:
        st.warning("La feuille sélectionnée ne contient pas de fixations valides.")
        return

    selection_key = f"{workbook_path}::{sheet_name}"
    _init_page0_state(selection_key)

    max_idx = len(fixations) - 1
    st.session_state.page0_frame_idx = min(st.session_state.page0_frame_idx, max_idx)

    with left_col:
        play_col, reset_col = st.columns(2)

        if play_col.button("Play", use_container_width=True, key="page0_play"):
            st.session_state.page0_is_playing = True
            st.session_state.page0_last_rerun_time = time.time()
            st.session_state.page0_play_progress = 0.0

        if reset_col.button("Reset", use_container_width=True, key="page0_reset"):
            st.session_state.page0_frame_idx = 0
            st.session_state.page0_is_playing = False
            st.session_state.page0_last_rerun_time = None
            st.session_state.page0_play_progress = 0.0

        manual_idx = st.slider(
            "Fixation index",
            min_value=0,
            max_value=max_idx,
            value=int(st.session_state.page0_frame_idx),
            step=1,
            key="page0_fixation_idx",
        )

        if not st.session_state.page0_is_playing:
            st.session_state.page0_frame_idx = int(manual_idx)

        st.caption(f"Fixation affichée : {st.session_state.page0_frame_idx + 1}/{len(fixations)}")

    current_idx = int(st.session_state.page0_frame_idx)

    visible_fixations = fixations.iloc[: current_idx + 1]
    x_coords, y_coords = aoi_to_coordinates(visible_fixations["AOI"])

    current_aoi = int(fixations.iloc[current_idx]["AOI"])
    current_point = (x_coords[-1], y_coords[-1])

    stage0_params = Stage0Params(
        lambda_stat=lambda_stat,
        lambda_br=lambda_br,
        sigma_comp=sigma_comp,
    )
    score0_params = Score0Params(
        sigma_score=sigma_score,
    )

    step = stage_0_step(
        game=sheet_name,
        x_t=float(current_aoi),
        params=stage0_params,
    )

    score_result = stage0_br_count_distribution(
        fixations=fixations,
        game=sheet_name,
        stage0_params=stage0_params,
        score0_params=score0_params,
    )

    meta = get_game_metadata(sheet_name)

    with center_col:
        st.markdown(
            f"**Jeu :** `{sheet_name}` · "
            f"**Type :** `{meta['game_type']}` · "
            f"**Règle :** `{meta['rule']}`"
        )
        st.markdown(
            f"**AOI actuelle :** `{current_aoi}` · "
            f"**T_g(x_t) :** `{float(step['T_g']):.3f}` · "
            f"**sigma_comp :** `{float(step['sigma_comp']):.3f}` · "
            f"**sigma_score :** `{float(score0_params.sigma_score):.3f}`"
        )
        st.markdown(
            f"**lambda_stat normalisé :** `{float(step['lambda_stat']):.3f}` · "
            f"**lambda_br normalisé :** `{float(step['lambda_br']):.3f}`"
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Target",
                "BR computationnelle",
                "Carte Stage 0",
                "Score BR",
            ]
        )

        with tab1:
            target_vector = np.zeros(AOI_COUNT, dtype=float)
            target_idx = int(round(float(step["T_g"]))) - 1
            target_idx = max(0, min(AOI_COUNT - 1, target_idx))
            target_vector[target_idx] = 1.0

            fig = _plot_heatmap(
                target_vector,
                title="Cible stratégique",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
                cmap="magma",
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            fig = _plot_heatmap(
                np.asarray(step["BR_comp_t"], dtype=float),
                title="BR computational",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            col_a, col_b = st.columns(2)

            with col_a:
                fig = _plot_heatmap(
                    np.asarray(step["S_stage0_t"], dtype=float),
                    title="S_stage0_t",
                    x_coords=x_coords,
                    y_coords=y_coords,
                    current_point=current_point,
                )
                st.pyplot(fig)
                plt.close(fig)

            with col_b:
                fig = _plot_heatmap(
                    np.asarray(step["q_t"], dtype=float),
                    title="q_t (probabilité)",
                    x_coords=x_coords,
                    y_coords=y_coords,
                    current_point=current_point,
                )
                st.pyplot(fig)
                plt.close(fig)

        with tab4:
            summary = score_result["summary"]
            dist = np.asarray(score_result["distribution"], dtype=float)
            br_df = score_result["transition_df"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("E[N_BR]", f"{float(summary['expected_count']):.3f}")
            col2.metric("Mode", f"{int(summary['mode'])}")
            col3.metric("Médiane", f"{int(summary['median'])}")
            col4.metric("P(N≥1)", f"{100 * float(summary['prob_at_least_one']):.1f}%")

            col5, col6, col7 = st.columns(3)
            col5.metric("Var[N_BR]", f"{float(summary['variance']):.3f}")
            col6.metric("Mean p_t", f"{float(summary['mean_p_t']):.3f}")
            col7.metric("Mean dist. BR", f"{float(summary['mean_distance_to_br']):.3f}")

            st.metric("Mean score entropy", f"{float(summary['mean_score_entropy']):.3f}")

            st.markdown("**Distribution du nombre de BR**")
            fig_dist, ax_dist = plt.subplots(figsize=(7.0, 3.5))
            ax_dist.bar(np.arange(len(dist)), dist)
            ax_dist.set_xlabel("Nombre de BR")
            ax_dist.set_ylabel("Probabilité")
            ax_dist.set_title("Distribution Poisson-binômiale")
            st.pyplot(fig_dist)
            plt.close(fig_dist)

            st.markdown("**Transitions locales**")
            display_cols = [
                c for c in [
                    "t",
                    "x_t",
                    "x_t_plus_1",
                    "T_g",
                    "distance_to_br",
                    "p_t",
                    "q_next_exact",
                    "q_max",
                    "score_entropy",
                ] if c in br_df.columns
            ]
            st.dataframe(br_df[display_cols], use_container_width=True, height=260)

    if st.session_state.page0_is_playing:
        display_step_s = 0.05

        now = time.time()
        last_t = st.session_state.page0_last_rerun_time
        real_delta_s = min(now - last_t, 0.5) if last_t is not None else 0.0
        st.session_state.page0_last_rerun_time = now

        st.session_state.page0_play_progress += real_delta_s * float(speed)
        frame_advance = int(st.session_state.page0_play_progress)

        if frame_advance > 0:
            st.session_state.page0_play_progress -= frame_advance
            new_idx = min(max_idx, int(st.session_state.page0_frame_idx) + frame_advance)
            st.session_state.page0_frame_idx = new_idx

            if new_idx >= max_idx:
                st.session_state.page0_is_playing = False

        time.sleep(display_step_s)
        st.rerun()