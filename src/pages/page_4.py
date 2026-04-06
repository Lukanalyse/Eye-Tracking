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
from src.saliency.dynamique_1 import (
    DynamicStage1Params,
    dynamic_stage1_step,
)
from src.saliency.static_salience import get_game_metadata


TIME_STEP_MS = 9


def _minmax_for_display(values: np.ndarray) -> np.ndarray:
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    return (values - v_min) / (v_max - v_min + 1e-12)


def _init_page_state(selection_key: str) -> None:
    if st.session_state.get("page4_selection_key") != selection_key:
        st.session_state.page4_selection_key = selection_key
        st.session_state.page4_frame_idx = 0
        st.session_state.page4_is_playing = False
        st.session_state.page4_elapsed_in_fixation = 0.0
        st.session_state.page4_last_rerun_time = None

    st.session_state.setdefault("page4_frame_idx", 0)
    st.session_state.setdefault("page4_is_playing", False)
    st.session_state.setdefault("page4_elapsed_in_fixation", 0.0)
    st.session_state.setdefault("page4_last_rerun_time", None)


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
    ax.set_xlabel("Colonnes")
    ax.set_ylabel("Lignes")
    ax.set_title(title)

    return fig


def _display_basic_info(
    sheet_name: str,
    current_aoi: int,
    tau_t_ms: float,
    u_ms: float,
    cumulative_time_ms: float,
    total_time_ms: float,
    dyn_step: dict,
    params: DynamicStage1Params,
) -> None:
    meta = get_game_metadata(sheet_name)

    st.markdown(
        f"**Jeu :** `{sheet_name}` · "
        f"**Type :** `{meta['game_type']}` · "
        f"**Règle :** `{meta['rule']}`"
    )
    st.markdown(
        f"**AOI actuelle :** `{current_aoi}` · "
        f"**Cible stratégique T_g(x_t) :** `{float(dyn_step['T_g']):.2f}` · "
        f"**rho :** `{float(dyn_step['rho']):.3f}` · "
        f"**sigma_comp_t :** `{float(dyn_step['sigma_comp_t']):.3f}`"
    )
    st.markdown(
        f"**lambda_stat :** `{params.lambda_stat:.3f}` · "
        f"**lambda_dyn :** `{params.lambda_dyn:.3f}` · "
        f"**sigma_space :** `{params.sigma_space:.3f}` · "
        f"**kappa_game :** `{params.kappa_game:.3f}` · "
        f"**tau_rho_ms :** `{params.tau_rho_ms:.1f}`"
    )

    fix_progress = min(u_ms / max(tau_t_ms, 1.0), 1.0)
    global_progress = min(cumulative_time_ms / max(total_time_ms, 1.0), 1.0)

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Fixation courante : `{u_ms:.0f} / {tau_t_ms:.0f} ms`")
        st.progress(fix_progress)
    with col2:
        st.caption(f"Progression globale : `{cumulative_time_ms:.0f} / {total_time_ms:.0f} ms`")
        st.progress(global_progress)


def show_page_4() -> None:
    st.header("Modélisation dynamique - Stage 1")
    st.caption(
        "Étape 1 : baseline uniforme + cible stratégique + blur computationnel + blur spatial."
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

    left_col, center_col = st.columns([1.1, 2.2])

    with left_col:
        st.subheader("Sélection")
        player_name = st.selectbox("Joueur", options=list(player_files.keys()), key="page4_player")
        workbook_path = player_files[player_name]

        workbook = load_workbook_sheets(workbook_path)
        sheet_names = [name for name in workbook if is_supported_game_sheet(name)]

        if not sheet_names:
            st.warning("Aucune feuille GAME1..GAME6 exploitable dans ce fichier.")
            return

        sheet_name = st.selectbox(
            "Jeu (sheet)",
            options=sorted(sheet_names, key=extract_game_number),
            key="page4_sheet",
        )

        st.markdown("**Lecture**")
        beta_enabled = st.checkbox("Activer le softmax final", value=True, key="page4_beta_enabled")
        beta_value = st.slider("beta", 0.0001, 0.5, 0.01, 0.0001, format="%.4f", key="page4_beta")
        speed = st.slider("Vitesse de lecture", 0.1, 20.0, 1.0, 0.1, key="page4_speed")

        st.markdown("**Paramètres du modèle**")
        lambda_stat = st.slider("lambda_stat", 0.0, 3.0, 0.3, 0.05, key="page4_lambda_stat")
        lambda_dyn = st.slider("lambda_dyn", 0.0, 5.0, 2.0, 0.05, key="page4_lambda_dyn")
        sigma_comp_min = st.slider("sigma_comp_min", 0.2, 15.0, 2.0, 0.1, key="page4_sigma_comp_min")
        sigma_comp_max = st.slider("sigma_comp_max", 0.2, 25.0, 10.0, 0.1, key="page4_sigma_comp_max")
        sigma_space = st.slider("sigma_space", 0.1, 8.0, 1.5, 0.1, key="page4_sigma_space")
        kappa_game = st.slider("kappa_game", 1.0, 100.0, 20.0, 1.0, key="page4_kappa_game")
        tau_rho_ms = st.slider("tau_rho_ms", 10.0, 500.0, 120.0, 5.0, key="page4_tau_rho_ms")

        with st.expander("Aide rapide", expanded=False):
            st.markdown(
                """
- `sigma_comp_min` : dispersion computationnelle minimale atteinte en fin de jeu.
- `sigma_comp_max` : dispersion computationnelle au début du jeu.
- `kappa_game` : vitesse à laquelle la dispersion computationnelle diminue.
- `sigma_space` : diffusion spatiale sur la grille 10×10.
- `tau_rho_ms` : constante de temps de l'activation intra-fixation.
- `rho(u) = 1 - exp(-u / tau_rho)` : activation croissante avec saturation.
"""
            )

    fixations = workbook[sheet_name]
    if fixations.empty:
        st.warning("La feuille sélectionnée ne contient pas de fixations valides.")
        return

    selection_key = f"{workbook_path}::{sheet_name}"
    _init_page_state(selection_key)

    max_idx = len(fixations) - 1
    st.session_state.page4_frame_idx = min(st.session_state.page4_frame_idx, max_idx)

    with left_col:
        play_col, reset_col = st.columns(2)
        if play_col.button("Play", use_container_width=True, key="page4_play"):
            st.session_state.page4_is_playing = True
            st.session_state.page4_last_rerun_time = time.time()

        if reset_col.button("Reset", use_container_width=True, key="page4_reset"):
            st.session_state.page4_frame_idx = 0
            st.session_state.page4_is_playing = False
            st.session_state.page4_elapsed_in_fixation = 0.0
            st.session_state.page4_last_rerun_time = None

        st.caption(f"Fixation affichée : {st.session_state.page4_frame_idx + 1}/{len(fixations)}")

    current_idx = int(st.session_state.page4_frame_idx)
    visible_fixations = fixations.iloc[: current_idx + 1]

    x_coords, y_coords = aoi_to_coordinates(visible_fixations["AOI"])
    current_aoi = int(fixations.iloc[current_idx]["AOI"])
    tau_t_ms = float(fixations.iloc[current_idx]["Time"])

    elapsed_ms_raw = min(float(st.session_state.page4_elapsed_in_fixation), tau_t_ms)
    u_ms = min(float((elapsed_ms_raw // TIME_STEP_MS) * TIME_STEP_MS), tau_t_ms)

    params = DynamicStage1Params(
        lambda_stat=lambda_stat,
        lambda_dyn=lambda_dyn,
        sigma_comp_min=sigma_comp_min,
        sigma_comp_max=sigma_comp_max,
        sigma_space=sigma_space,
        kappa_game=kappa_game,
        tau_rho_ms=tau_rho_ms,
        beta=beta_value if beta_enabled else None,
    )

    dyn_step = dynamic_stage1_step(
        game=sheet_name,
        x_t=current_aoi,
        u_ms=u_ms,
        fixation_index=current_idx,
        params=params,
    )

    completed_time_ms = float(fixations.iloc[:current_idx]["Time"].sum())
    cumulative_time_ms = completed_time_ms + elapsed_ms_raw
    total_time_ms = float(fixations["Time"].sum())

    current_point = (x_coords[-1], y_coords[-1])

    with center_col:
        _display_basic_info(
            sheet_name=sheet_name,
            current_aoi=current_aoi,
            tau_t_ms=tau_t_ms,
            u_ms=u_ms,
            cumulative_time_ms=cumulative_time_ms,
            total_time_ms=total_time_ms,
            dyn_step=dyn_step,
            params=params,
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Target",
                "BR computationnelle",
                "BR spatiale",
                "Carte finale",
                "Global",
            ]
        )

        with tab1:
            st.subheader("Cible stratégique")
            st.write("Cet onglet permet de vérifier que la règle du jeu produit bien une cible stratégique cohérente.")
            st.metric("AOI observée x_t", current_aoi)
            st.metric("Cible stratégique T_g(x_t)", f"{float(dyn_step['T_g']):.3f}")
            st.metric("Fixation index", current_idx)
            st.metric("sigma_comp_t", f"{float(dyn_step['sigma_comp_t']):.3f}")

            target_vector = np.zeros(AOI_COUNT, dtype=float)
            target_idx = int(round(float(dyn_step["T_g"]))) - 1
            target_idx = max(0, min(AOI_COUNT - 1, target_idx))
            target_vector[target_idx] = 1.0

            fig = _plot_heatmap(
                target_vector,
                title="Position de la cible stratégique (projection arrondie)",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
                cmap="magma",
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            st.subheader("Best Response computationnelle")
            st.write("Carte gaussienne sur les valeurs 1..100 autour de T_g(x_t), avant diffusion spatiale.")

            fig = _plot_heatmap(
                np.asarray(dyn_step["BR_comp_t"], dtype=float),
                title="BR computationnelle",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
            )
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Voir les 15 plus fortes valeurs"):
                br_comp = np.asarray(dyn_step["BR_comp_t"], dtype=float)
                top_idx = np.argsort(br_comp)[::-1][:15]
                top_table = {
                    "AOI": (top_idx + 1).tolist(),
                    "BR_comp_t": [float(br_comp[i]) for i in top_idx],
                }
                st.dataframe(top_table, use_container_width=True)

        with tab3:
            st.subheader("Best Response après blur spatial")
            st.write("La carte computationnelle est ensuite diffusée spatialement sur la grille 10×10.")

            fig = _plot_heatmap(
                np.asarray(dyn_step["BR_space_t"], dtype=float),
                title="BR spatiale",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab4:
            st.subheader("Carte dynamique finale")
            st.write("Carte finale : baseline uniforme + composante dynamique pondérée par rho.")

            fig = _plot_heatmap(
                np.asarray(dyn_step["S_dyn_t"], dtype=float),
                title="S_dyn_t",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
            )
            st.pyplot(fig)
            plt.close(fig)

            if "q_t" in dyn_step:
                st.markdown("**Softmax final**")
                fig_q = _plot_heatmap(
                    np.asarray(dyn_step["q_t"], dtype=float),
                    title="q_t",
                    x_coords=x_coords,
                    y_coords=y_coords,
                    current_point=current_point,
                    cmap="viridis",
                )
                st.pyplot(fig_q)
                plt.close(fig_q)

        with tab5:
            st.subheader("Diagnostic global")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("rho", f"{float(dyn_step['rho']):.4f}")
            col2.metric("sigma_comp_t", f"{float(dyn_step['sigma_comp_t']):.4f}")
            col3.metric("Temps cumulé", f"{cumulative_time_ms:.0f} ms")
            col4.metric("Fixations", f"{current_idx + 1}/{len(fixations)}")

            st.caption("Trajectoire AOI observée jusqu'à la fixation courante")
            ordered_aoi = visible_fixations["AOI"].astype(str).tolist()
            if len(ordered_aoi) > 50:
                st.code(" -> ".join(ordered_aoi[:50]) + " -> ...")
            else:
                st.code(" -> ".join(ordered_aoi))

            with st.expander("Tableau des fixations"):
                st.dataframe(fixations, use_container_width=True, height=250)

            with st.expander("Grille AOI 10x10"):
                st.dataframe(
                    aoi_to_matrix(np.arange(1, AOI_COUNT + 1)),
                    use_container_width=True,
                    height=240,
                )

    if st.session_state.page4_is_playing:
        display_step_s = 0.04

        now = time.time()
        last_t = st.session_state.page4_last_rerun_time
        real_delta_ms = min((now - last_t) * 1000.0, 500.0) if last_t is not None else 0.0
        st.session_state.page4_last_rerun_time = now

        sim_delta_ms = real_delta_ms * float(speed)
        new_elapsed = float(st.session_state.page4_elapsed_in_fixation) + sim_delta_ms

        frame = int(st.session_state.page4_frame_idx)
        ended = False

        while True:
            fix_dur = max(float(fixations.iloc[frame]["Time"]), 1.0)
            if new_elapsed < fix_dur:
                break
            new_elapsed -= fix_dur
            if frame < max_idx:
                frame += 1
            else:
                new_elapsed = fix_dur
                ended = True
                break

        st.session_state.page4_frame_idx = frame
        st.session_state.page4_elapsed_in_fixation = new_elapsed

        if ended:
            st.session_state.page4_is_playing = False

        time.sleep(display_step_s)
        st.rerun()