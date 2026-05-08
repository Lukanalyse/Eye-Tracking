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
from src.saliency.static_salience import get_game_metadata
from src.stage_1.stage_1 import Stage1Params, stage_1_step
from src.stage_1.score_adapter import stage1_br_count_distribution
from src.stage_0.score_0 import Score0Params


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
    previous_point: tuple[float, float] | None = None,
    cmap: str = "magma",
) -> plt.Figure:
    matrix = aoi_to_matrix(_minmax_for_display(values))

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    sns.heatmap(matrix, cmap=cmap, square=True, cbar=True, ax=ax)

    if x_coords is not None and y_coords is not None and len(x_coords) > 0:
        ax.plot(x_coords, y_coords, color="white", linewidth=1.2, alpha=0.95)

    if previous_point is not None:
        ax.scatter(
            previous_point[0],
            previous_point[1],
            color="cyan",
            s=65,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label="Previous fixation",
        )

    if current_point is not None:
        ax.scatter(
            current_point[0],
            current_point[1],
            color="lightgreen",
            s=70,
            edgecolor="black",
            linewidth=0.5,
            zorder=6,
            label="Current fixation",
        )

    if previous_point is not None or current_point is not None:
        ax.legend(loc="upper right", fontsize=8)

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


def _plot_br_distribution(dist: np.ndarray, title: str) -> plt.Figure:
    dist = np.asarray(dist, dtype=float)

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.bar(np.arange(len(dist)), dist)

    ax.set_xlabel("Number of BR episodes")
    ax.set_ylabel("Probability")
    ax.set_title(title)

    if len(dist) > 0:
        mode = int(np.argmax(dist))
        max_prob = float(np.max(dist))

        ax.axvline(mode, linestyle="--", linewidth=1.2)
        ax.text(
            mode,
            max_prob + 0.01,
            f"mode={mode}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return fig


def _init_page1_state(selection_key: str) -> None:
    if st.session_state.get("stage1_selection_key") != selection_key:
        st.session_state.stage1_selection_key = selection_key
        st.session_state.stage1_frame_idx = 0
        st.session_state.stage1_fixation_idx = 0
        st.session_state.stage1_is_playing = False
        st.session_state.stage1_last_rerun_time = None
        st.session_state.stage1_play_progress = 0.0

    st.session_state.setdefault("stage1_frame_idx", 0)
    st.session_state.setdefault("stage1_fixation_idx", 0)
    st.session_state.setdefault("stage1_is_playing", False)
    st.session_state.setdefault("stage1_last_rerun_time", None)
    st.session_state.setdefault("stage1_play_progress", 0.0)


def show_stage1_page() -> None:
    st.header("Stage 1 - Computational BR + Spatial Blur")
    st.caption(
        "Stage 1 model: computational BR followed by Gaussian spatial diffusion over the grid. "
        "Default calibrated values: sigma_comp=2.0, sigma_space=1.5, sigma_score=8."
    )

    project_root = Path(__file__).resolve().parents[2]
    data_dir, output_dir = get_data_dirs()
    player_files = discover_player_files(str(project_root))

    if not player_files:
        st.warning("No player Excel file was found.")
        st.info(
            f"Scanned folders:\n- {project_root}\n- {output_dir}\n- {data_dir}\n\n"
            "Place the player Excel files in one of these folders."
        )
        return

    left_col, center_col = st.columns([1.15, 2.25])

    with left_col:
        st.subheader("Selection")
        player_name = st.selectbox(
            "Player",
            options=list(player_files.keys()),
            key="stage1_player",
        )
        workbook_path = player_files[player_name]

        workbook = load_workbook_sheets(workbook_path)
        sheet_names = [name for name in workbook if is_supported_game_sheet(name)]

        if not sheet_names:
            st.warning("No valid GAME1..GAME6 sheet was found in this file.")
            return

        sheet_name = st.selectbox(
            "Game (sheet)",
            options=sorted(sheet_names, key=extract_game_number),
            key="stage1_sheet",
        )

        st.markdown("**Stage 1 Parameters**")
        lambda_stat = st.slider(
            "lambda_stat",
            0.0,
            1.0,
            0.0,
            0.01,
            key="stage1_lambda_stat",
        )
        lambda_br = st.slider(
            "lambda_br",
            0.0,
            1.0,
            1.0,
            0.01,
            key="stage1_lambda_br",
        )
        sigma_comp = st.slider(
            "sigma_comp",
            0.2,
            25.0,
            2.0,
            0.1,
            key="stage1_sigma_comp",
        )
        sigma_space = st.slider(
            "sigma_space",
            0.2,
            8.0,
            1.5,
            0.1,
            key="stage1_sigma_space",
        )

        st.markdown("**Stage 1 Score Parameters**")
        sigma_score = st.slider(
            "sigma_score",
            0.5,
            20.0,
            8.0,
            0.1,
            key="stage1_sigma_score",
        )
        use_relative_overlap = st.checkbox(
            "Use relative overlap score",
            value=True,
            key="stage1_use_relative_overlap",
        )

        st.markdown("**Dynamic Playback**")
        speed = st.slider(
            "Playback speed",
            0.05,
            5.0,
            0.5,
            0.05,
            key="stage1_speed",
        )

    fixations = workbook[sheet_name]

    if fixations.empty:
        st.warning("The selected sheet does not contain valid fixations.")
        return

    selection_key = f"{workbook_path}::{sheet_name}"
    _init_page1_state(selection_key)

    max_idx = len(fixations) - 1
    st.session_state.stage1_frame_idx = min(st.session_state.stage1_frame_idx, max_idx)

    with left_col:
        play_col, pause_col, reset_col = st.columns(3)

        if play_col.button("Play", width="stretch", key="stage1_play"):
            st.session_state.stage1_is_playing = True
            st.session_state.stage1_last_rerun_time = time.time()
            st.session_state.stage1_play_progress = 0.0

        if pause_col.button("Pause", width="stretch", key="stage1_pause"):
            st.session_state.stage1_is_playing = False
            st.session_state.stage1_last_rerun_time = None
            st.session_state.stage1_play_progress = 0.0
            st.session_state.stage1_fixation_idx = int(st.session_state.stage1_frame_idx)

        if reset_col.button("Reset", width="stretch", key="stage1_reset"):
            st.session_state.stage1_frame_idx = 0
            st.session_state.stage1_is_playing = False
            st.session_state.stage1_last_rerun_time = None
            st.session_state.stage1_play_progress = 0.0
            st.session_state.stage1_fixation_idx = 0

        # Initialize slider state only once
        if "stage1_fixation_idx" not in st.session_state:
            st.session_state.stage1_fixation_idx = int(st.session_state.stage1_frame_idx)

        st.slider(
            "Fixation index",
            min_value=0,
            max_value=max_idx,
            step=1,
            key="stage1_fixation_idx",
        )

        if not st.session_state.stage1_is_playing:
            st.session_state.stage1_frame_idx = int(st.session_state.stage1_fixation_idx)

        if st.session_state.stage1_is_playing:
            st.caption("Status: playing")
        else:
            st.caption("Status: paused")

        st.caption(
            f"Displayed fixation: {st.session_state.stage1_frame_idx + 1}/{len(fixations)}"
        )

    current_idx = int(st.session_state.stage1_frame_idx)

    visible_fixations = fixations.iloc[: current_idx + 1]
    x_coords, y_coords = aoi_to_coordinates(visible_fixations["AOI"])

    current_aoi = int(fixations.iloc[current_idx]["AOI"])
    current_point = (x_coords[-1], y_coords[-1])

    previous_aoi = None
    previous_point = None

    if current_idx >= 1:
        previous_aoi = int(fixations.iloc[current_idx - 1]["AOI"])
        previous_visible_fixations = fixations.iloc[:current_idx]
        prev_x_coords, prev_y_coords = aoi_to_coordinates(previous_visible_fixations["AOI"])
        previous_point = (prev_x_coords[-1], prev_y_coords[-1])

    stage1_params = Stage1Params(
        lambda_stat=lambda_stat,
        lambda_br=lambda_br,
        sigma_comp=sigma_comp,
        sigma_space=sigma_space,
    )

    score_params = Score0Params(
        sigma_score=sigma_score,
        use_relative_overlap=use_relative_overlap,
    )

    step = stage_1_step(
        game=sheet_name,
        x_t=float(current_aoi),
        params=stage1_params,
    )

    previous_step = None

    if previous_aoi is not None:
        previous_step = stage_1_step(
            game=sheet_name,
            x_t=float(previous_aoi),
            params=stage1_params,
        )

    # Dynamic BR distribution: only up to the current fixation.
    if len(visible_fixations) >= 2:
        score_result_dynamic = stage1_br_count_distribution(
            fixations=visible_fixations,
            game=sheet_name,
            stage1_params=stage1_params,
            score_params=score_params,
        )
    else:
        score_result_dynamic = None

    # Full BR distribution: entire sequence.
    score_result_full = stage1_br_count_distribution(
        fixations=fixations,
        game=sheet_name,
        stage1_params=stage1_params,
        score_params=score_params,
    )

    meta = get_game_metadata(sheet_name)

    with center_col:
        st.markdown(
            f"**Game:** `{sheet_name}` · "
            f"**Type:** `{meta['game_type']}` · "
            f"**Rule:** `{meta['rule']}`"
        )
        st.markdown(
            f"**Current AOI:** `{current_aoi}` · "
            f"**T_g(x_t):** `{float(step['T_g']):.3f}` · "
            f"**sigma_comp:** `{float(step['sigma_comp']):.3f}` · "
            f"**sigma_space:** `{float(step['sigma_space']):.3f}`"
        )
        st.markdown(
            f"**sigma_score:** `{float(score_params.sigma_score):.3f}` · "
            f"**Relative overlap:** `{score_params.use_relative_overlap}`"
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Target",
                "Computational BR",
                "Spatial BR",
                "Stage 1 Map",
                "BR Score",
            ]
        )

        with tab1:
            target_vector = np.zeros(AOI_COUNT, dtype=float)
            target_idx = int(round(float(step["T_g"]))) - 1
            target_idx = max(0, min(AOI_COUNT - 1, target_idx))
            target_vector[target_idx] = 1.0

            fig = _plot_heatmap(
                target_vector,
                title="Strategic Target",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
                previous_point=previous_point,
                cmap="magma",
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            fig = _plot_heatmap(
                np.asarray(step["BR_comp_t"], dtype=float),
                title="Computational BR",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
                previous_point=previous_point,
                cmap="magma",
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            fig = _plot_heatmap(
                np.asarray(step["BR_space_t"], dtype=float),
                title="Spatial BR",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
                previous_point=previous_point,
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab4:
            st.markdown("### Dynamic Stage 1 Map and BR Distribution")

            main_col, side_col = st.columns([2.1, 1.2])

            with main_col:
                fig = _plot_heatmap(
                    np.asarray(step["q_t"], dtype=float),
                    title=f"Current Stage 1 map q_t | fixation {current_idx + 1}",
                    x_coords=x_coords,
                    y_coords=y_coords,
                    current_point=current_point,
                    previous_point=previous_point,
                    cmap="magma",
                )
                st.pyplot(fig)
                plt.close(fig)

                if previous_aoi is not None:
                    st.caption(
                        f"Previous AOI: {previous_aoi} → Current AOI: {current_aoi}. "
                        "Orange point = previous fixation. Red point = current fixation."
                    )

            with side_col:
                st.markdown("**Previous salience map**")

                if previous_step is not None:
                    fig_prev = _plot_heatmap(
                        np.asarray(previous_step["q_t"], dtype=float),
                        title=f"Previous map q_(t-1) | AOI {previous_aoi}",
                        x_coords=None,
                        y_coords=None,
                        current_point=current_point,
                        previous_point=previous_point,
                        cmap="magma",
                    )
                    st.pyplot(fig_prev)
                    plt.close(fig_prev)

                    st.caption(
                        "This small map shows the salience generated by the previous fixation. "
                        "It helps assess whether the player moved into a previously salient region."
                    )
                else:
                    st.info("No previous transition yet.")
                st.markdown("---")
                st.markdown("**Dynamic BR distribution**")

                if score_result_dynamic is not None:
                    dynamic_summary = score_result_dynamic["summary"]
                    dynamic_dist = np.asarray(score_result_dynamic["distribution"], dtype=float)

                    n_transitions_so_far = max(current_idx, 1)
                    br_rate = float(dynamic_summary["expected_count"]) / n_transitions_so_far

                    last_p = float(score_result_dynamic["transition_df"]["p_t"].iloc[-1])

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric(
                            "E[N_BR so far]",
                            f"{float(dynamic_summary['expected_count']):.3f}",
                        )

                    with metric_col2:
                        st.metric(
                            "Rate",
                            f"{100 * br_rate:.2f}%",
                        )

                    with metric_col3:
                        st.metric(
                            "Last transition p_t",
                            f"{100 * last_p:.2f}%",
                        )

                    metric_col4, metric_col5 = st.columns(2)

                    with metric_col4:
                        st.metric(
                            "Mode so far",
                            f"{int(dynamic_summary['mode'])}",
                        )

                    with metric_col5:
                        st.metric(
                            "P(N≥1 so far)",
                            f"{100 * float(dynamic_summary['prob_at_least_one']):.1f}%",
                        )

                    fig_dyn = _plot_br_distribution(
                        dynamic_dist,
                        title="BR distribution so far",
                    )
                    st.pyplot(fig_dyn)
                    plt.close(fig_dyn)

                else:
                    st.info("At least two fixations are needed to compute a BR transition.")



        with tab5:
            summary = score_result_full["summary"]
            dist = np.asarray(score_result_full["distribution"], dtype=float)
            br_df = score_result_full["transition_df"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("E[N_BR]", f"{float(summary['expected_count']):.3f}")
            col2.metric("Mode", f"{int(summary['mode'])}")
            col3.metric("Median", f"{int(summary['median'])}")
            col4.metric("P(N≥1)", f"{100 * float(summary['prob_at_least_one']):.1f}%")

            col5, col6 = st.columns(2)
            col5.metric("Var[N_BR]", f"{float(summary['variance']):.3f}")
            col6.metric("P(N=0)", f"{100 * float(dist[0]):.1f}%")

            st.markdown("**Distribution of the Number of BR Episodes**")
            fig_dist, ax_dist = plt.subplots(figsize=(7.0, 3.5))
            ax_dist.bar(np.arange(len(dist)), dist)
            ax_dist.set_xlabel("Number of BR Episodes")
            ax_dist.set_ylabel("Probability")
            ax_dist.set_title("Poisson-Binomial Distribution - Full Sequence")
            st.pyplot(fig_dist)
            plt.close(fig_dist)

            st.markdown("**Local Transition Probabilities**")
            st.dataframe(br_df, width="stretch", height=260)

    if st.session_state.stage1_is_playing:
        display_step_s = 0.05

        now = time.time()
        last_t = st.session_state.stage1_last_rerun_time
        real_delta_s = min(now - last_t, 0.5) if last_t is not None else 0.0
        st.session_state.stage1_last_rerun_time = now

        st.session_state.stage1_play_progress += real_delta_s * float(speed)
        frame_advance = int(st.session_state.stage1_play_progress)

        if frame_advance > 0:
            st.session_state.stage1_play_progress -= frame_advance
            new_idx = min(max_idx, int(st.session_state.stage1_frame_idx) + frame_advance)
            st.session_state.stage1_frame_idx = new_idx

            if new_idx >= max_idx:
                st.session_state.stage1_is_playing = False

        time.sleep(display_step_s)
        st.rerun()