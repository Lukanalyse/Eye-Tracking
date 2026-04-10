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
from src.saliency.dynamique_2 import (
    DynamicStage2Params,
    dynamic_stage2_step,
)
from src.saliency.static_salience import get_game_metadata


TIME_STEP_MS = 9


def _minmax_for_display(values: np.ndarray) -> np.ndarray:
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    return (values - v_min) / (v_max - v_min + 1e-12)


def _init_page_state(selection_key: str) -> None:
    if st.session_state.get("page5_selection_key") != selection_key:
        st.session_state.page5_selection_key = selection_key
        st.session_state.page5_frame_idx = 0
        st.session_state.page5_is_playing = False
        st.session_state.page5_elapsed_in_fixation = 0.0
        st.session_state.page5_last_rerun_time = None

    st.session_state.setdefault("page5_frame_idx", 0)
    st.session_state.setdefault("page5_is_playing", False)
    st.session_state.setdefault("page5_elapsed_in_fixation", 0.0)
    st.session_state.setdefault("page5_last_rerun_time", None)


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
    params: DynamicStage2Params,
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
        f"**kappa_game :** `{params.kappa_game:.3f}`"
    )
    st.markdown(
        f"**memory_depth :** `{params.memory_depth}` · "
        f"**memory_decay :** `{params.memory_decay:.3f}` · "
        f"**tau_memory_ms :** `{params.tau_memory_ms:.1f}` · "
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


def _build_previous_memory_inputs(
    fixations,
    sheet_name: str,
    current_idx: int,
    params: DynamicStage2Params,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Rebuild previous BR_space maps and associated elapsed times up to current_idx.
    Output order:
        [BR_{t-1}^{space}, BR_{t-2}^{space}, ...]
    """
    if current_idx <= 0:
        return [], []

    previous_br_space_maps: list[np.ndarray] = []
    previous_delta_times_ms: list[float] = []

    for idx in range(current_idx):
        current_aoi = float(fixations.iloc[idx]["AOI"])
        tau_t_ms = float(fixations.iloc[idx]["Time"])

        step_result = dynamic_stage2_step(
            game=sheet_name,
            x_t=current_aoi,
            u_ms=tau_t_ms,
            fixation_index=idx,
            params=params,
            previous_br_space_maps=previous_br_space_maps,
            previous_delta_times_ms=previous_delta_times_ms,
        )

        current_br_space = np.asarray(step_result["BR_space_t"], dtype=float)

        previous_br_space_maps.insert(0, current_br_space)
        previous_delta_times_ms.insert(0, 0.0)

        for j in range(len(previous_delta_times_ms)):
            previous_delta_times_ms[j] += tau_t_ms

        if len(previous_br_space_maps) > params.memory_depth:
            previous_br_space_maps = previous_br_space_maps[: params.memory_depth]
            previous_delta_times_ms = previous_delta_times_ms[: params.memory_depth]

    return previous_br_space_maps, previous_delta_times_ms

def _plot_small_heatmap(
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
) -> plt.Figure:
    matrix = aoi_to_matrix(_minmax_for_display(values))

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    sns.heatmap(matrix, cmap=cmap, square=True, cbar=True, ax=ax, vmin=0.0, vmax=1.0)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, 0)
    ax.set_xticks(np.arange(GRID_SIZE) + 0.5)
    ax.set_xticklabels(np.arange(1, GRID_SIZE + 1), fontsize=6)
    ax.set_yticks(np.arange(GRID_SIZE) + 0.5)
    ax.set_yticklabels(np.arange(0, GRID_SIZE), fontsize=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=8)

    return fig

def show_page_5() -> None:
    st.header("Modélisation dynamique - Stage 2 (mémoire locale)")
    st.caption(
        "Étape 2 : baseline uniforme + cible stratégique + blur computationnel + blur spatial + mémoire locale."
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
        player_name = st.selectbox("Joueur", options=list(player_files.keys()), key="page5_player")
        workbook_path = player_files[player_name]

        workbook = load_workbook_sheets(workbook_path)
        sheet_names = [name for name in workbook if is_supported_game_sheet(name)]

        if not sheet_names:
            st.warning("Aucune feuille GAME1..GAME6 exploitable dans ce fichier.")
            return

        sheet_name = st.selectbox(
            "Jeu (sheet)",
            options=sorted(sheet_names, key=extract_game_number),
            key="page5_sheet",
        )

        st.markdown("**Lecture**")
        beta_enabled = st.checkbox("Activer le softmax final", value=True, key="page5_beta_enabled")
        beta_value = st.slider("beta", 0.0001, 0.5, 0.01, 0.0001, format="%.4f", key="page5_beta")
        speed = st.slider("Vitesse de lecture", 0.1, 20.0, 1.0, 0.1, key="page5_speed")

        st.markdown("**Paramètres du modèle**")
        lambda_stat = st.slider("lambda_stat", 0.0, 3.0, 0.3, 0.05, key="page5_lambda_stat")
        lambda_dyn = st.slider("lambda_dyn", 0.0, 5.0, 2.0, 0.05, key="page5_lambda_dyn")
        sigma_comp_inf = st.slider("sigma_comp_inf", 0.2, 25.0, 12.0, 0.1, key="page5_sigma_comp_inf")
        sigma_comp_amp = st.slider("sigma_comp_amp", 0.0, 25.0, 20.0, 0.1, key="page5_sigma_comp_amp")
        sigma_space = st.slider("sigma_space", 0.1, 8.0, 1.5, 0.1, key="page5_sigma_space")
        kappa_game = st.slider("kappa_game", 1.0, 100.0, 20.0, 1.0, key="page5_kappa_game")
        tau_rho_ms = st.slider("tau_rho_ms", 10.0, 500.0, 120.0, 5.0, key="page5_tau_rho_ms")

        st.markdown("**Mémoire locale**")
        memory_depth = st.slider("memory_depth", 0, 6, 3, 1, key="page5_memory_depth")
        memory_decay = st.slider("memory_decay", 0.0, 1.0, 0.7, 0.05, key="page5_memory_decay")
        tau_memory_ms = st.slider("tau_memory_ms", 10.0, 5000.0, 800.0, 10.0, key="page5_tau_memory_ms")

        with st.expander("Aide rapide", expanded=False):
            st.markdown(
                r"""
- `memory_depth` : nombre maximal de cartes passées conservées.
- `memory_decay` : oubli par transition, via \(\delta^j\).
- `tau_memory_ms` : oubli continu en temps, via \(\exp(-\Delta T / \tau_{\mathrm{mem}})\).
- `tau_rho_ms` : constante de temps de l'activation intra-fixation.
- Le modèle combine signal courant et mémoire locale.
"""
            )

    fixations = workbook[sheet_name]
    if fixations.empty:
        st.warning("La feuille sélectionnée ne contient pas de fixations valides.")
        return

    selection_key = f"{workbook_path}::{sheet_name}"
    _init_page_state(selection_key)

    max_idx = len(fixations) - 1
    st.session_state.page5_frame_idx = min(st.session_state.page5_frame_idx, max_idx)

    with left_col:
        play_col, reset_col = st.columns(2)
        if play_col.button("Play", use_container_width=True, key="page5_play"):
            st.session_state.page5_is_playing = True
            st.session_state.page5_last_rerun_time = time.time()

        if reset_col.button("Reset", use_container_width=True, key="page5_reset"):
            st.session_state.page5_frame_idx = 0
            st.session_state.page5_is_playing = False
            st.session_state.page5_elapsed_in_fixation = 0.0
            st.session_state.page5_last_rerun_time = None

        st.caption(f"Fixation affichée : {st.session_state.page5_frame_idx + 1}/{len(fixations)}")

    current_idx = int(st.session_state.page5_frame_idx)
    visible_fixations = fixations.iloc[: current_idx + 1]

    x_coords, y_coords = aoi_to_coordinates(visible_fixations["AOI"])
    current_aoi = int(fixations.iloc[current_idx]["AOI"])
    tau_t_ms = float(fixations.iloc[current_idx]["Time"])

    elapsed_ms_raw = min(float(st.session_state.page5_elapsed_in_fixation), tau_t_ms)
    u_ms = min(float((elapsed_ms_raw // TIME_STEP_MS) * TIME_STEP_MS), tau_t_ms)

    params = DynamicStage2Params(
        lambda_stat=lambda_stat,
        lambda_dyn=lambda_dyn,
        sigma_comp_inf=sigma_comp_inf,
        sigma_comp_amp=sigma_comp_amp,
        sigma_space=sigma_space,
        kappa_game=kappa_game,
        tau_rho_ms=tau_rho_ms,
        memory_depth=memory_depth,
        memory_decay=memory_decay,
        tau_memory_ms=tau_memory_ms,
        beta=beta_value if beta_enabled else None,
    )

    previous_br_space_maps, previous_delta_times_ms = _build_previous_memory_inputs(
        fixations=fixations,
        sheet_name=sheet_name,
        current_idx=current_idx,
        params=params,
    )

    dyn_step = dynamic_stage2_step(
        game=sheet_name,
        x_t=current_aoi,
        u_ms=u_ms,
        fixation_index=current_idx,
        params=params,
        previous_br_space_maps=previous_br_space_maps,
        previous_delta_times_ms=previous_delta_times_ms,
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

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Target",
                "Current BR",
                "Memory",
                "Combined Dynamic",
                "Final Map",
                "Diagnostics",
            ]
        )

        with tab1:
            st.subheader("Cible stratégique")
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
            st.subheader("Signal courant")
            st.write("Composante instantanée : BR computationnelle, BR spatiale, puis activation courante.")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                fig = _plot_heatmap(
                    np.asarray(dyn_step["BR_comp_t"], dtype=float),
                    title="BR computationnelle",
                    x_coords=x_coords,
                    y_coords=y_coords,
                    current_point=current_point,
                )
                st.pyplot(fig)
                plt.close(fig)

            with col_b:
                fig = _plot_heatmap(
                    np.asarray(dyn_step["BR_space_t"], dtype=float),
                    title="BR spatiale",
                    x_coords=x_coords,
                    y_coords=y_coords,
                    current_point=current_point,
                )
                st.pyplot(fig)
                plt.close(fig)

            with col_c:
                fig = _plot_heatmap(
                    np.asarray(dyn_step["current_dynamic_t"], dtype=float),
                    title="rho × BR_spatiale",
                    x_coords=x_coords,
                    y_coords=y_coords,
                    current_point=current_point,
                )
                st.pyplot(fig)
                plt.close(fig)

            st.metric("rho", f"{float(dyn_step['rho']):.4f}")

        with tab3:
            st.subheader("Mémoire locale")
            st.write("Somme pondérée des cartes passées selon les transitions et le temps écoulé.")

            fig = _plot_heatmap(
                np.asarray(dyn_step["memory_map_t"], dtype=float),
                title="Memory map (agrégée)",
                x_coords=x_coords,
                y_coords=y_coords,
                current_point=current_point,
            )
            st.pyplot(fig)
            plt.close(fig)

            trans_weights = np.asarray(dyn_step["transition_weights"], dtype=float)
            temporal_factors = np.asarray(dyn_step["temporal_factors"], dtype=float)
            effective_weights = np.asarray(dyn_step["effective_memory_weights"], dtype=float)

            if len(effective_weights) > 0:
                st.markdown("---")
                st.subheader("Décomposition de la mémoire par profondeur")

                n_cols = min(len(effective_weights) + 1, 4)
                cols = st.columns(n_cols)

                # Carte courante pondérée par rho
                with cols[0]:
                    current_weighted = np.asarray(dyn_step["current_dynamic_t"], dtype=float)
                    fig_curr = _plot_small_heatmap(
                        current_weighted,
                        title=f"Current\nρ={float(dyn_step['rho']):.3f}",
                    )
                    st.pyplot(fig_curr, use_container_width=True)
                    plt.close(fig_curr)

                # Cartes passées pondérées
                for j in range(len(effective_weights)):
                    col_idx = (j + 1) % n_cols
                    if col_idx == 0:
                        cols = st.columns(n_cols)
                        col_idx = 0

                    past_map = np.asarray(previous_br_space_maps[j], dtype=float)
                    weighted_past = effective_weights[j] * past_map

                    title = (
                        f"t-{j + 1}\n"
                        f"δ^j={trans_weights[j]:.3f}\n"
                        f"time={temporal_factors[j]:.3f}\n"
                        f"w={effective_weights[j]:.3f}"
                    )

                    with cols[col_idx]:
                        fig_past = _plot_small_heatmap(
                            weighted_past,
                            title=title,
                        )
                        st.pyplot(fig_past, use_container_width=True)
                        plt.close(fig_past)

                st.markdown("---")
                st.subheader("Tableau des poids mémoire")

                weights_table = {
                    "lag_j": list(range(1, len(effective_weights) + 1)),
                    "delta^j": [float(x) for x in trans_weights],
                    "exp(-DeltaT/tau_mem)": [float(x) for x in temporal_factors],
                    "poids_effectif": [float(x) for x in effective_weights],
                    "DeltaT_ms": [float(x) for x in previous_delta_times_ms[: len(effective_weights)]],
                }
                st.dataframe(weights_table, use_container_width=True)

            else:
                st.info("Pas encore de mémoire disponible à cette fixation.")

        with tab4:
            st.subheader("Composante dynamique combinée")
            st.write("Somme du signal courant et de la mémoire locale.")

            col_a, col_b = st.columns(2)

            with col_a:
                fig = _plot_heatmap(
                    np.asarray(dyn_step["current_dynamic_t"], dtype=float),
                    title="Current dynamic",
                    x_coords=x_coords,
                    y_coords=y_coords,
                    current_point=current_point,
                )
                st.pyplot(fig)
                plt.close(fig)

            with col_b:
                fig = _plot_heatmap(
                    np.asarray(dyn_step["combined_dynamic_t"], dtype=float),
                    title="Combined dynamic",
                    x_coords=x_coords,
                    y_coords=y_coords,
                    current_point=current_point,
                )
                st.pyplot(fig)
                plt.close(fig)

        with tab5:
            st.subheader("Carte finale")
            st.write("Carte finale : baseline uniforme + composante dynamique combinée.")

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

        with tab6:
            st.subheader("Diagnostics")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("rho", f"{float(dyn_step['rho']):.4f}")
            col2.metric("sigma_comp_t", f"{float(dyn_step['sigma_comp_t']):.4f}")
            col3.metric("Temps cumulé", f"{cumulative_time_ms:.0f} ms")
            col4.metric("Fixations", f"{current_idx + 1}/{len(fixations)}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Max current dynamic", f"{float(np.max(dyn_step['current_dynamic_t'])):.4f}")
            col6.metric("Max memory", f"{float(np.max(dyn_step['memory_map_t'])):.4f}")
            col7.metric("Max combined", f"{float(np.max(dyn_step['combined_dynamic_t'])):.4f}")
            col8.metric("Max final", f"{float(np.max(dyn_step['S_dyn_t'])):.4f}")

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

    if st.session_state.page5_is_playing:
        display_step_s = 0.04

        now = time.time()
        last_t = st.session_state.page5_last_rerun_time
        real_delta_ms = min((now - last_t) * 1000.0, 500.0) if last_t is not None else 0.0
        st.session_state.page5_last_rerun_time = now

        sim_delta_ms = real_delta_ms * float(speed)
        new_elapsed = float(st.session_state.page5_elapsed_in_fixation) + sim_delta_ms

        frame = int(st.session_state.page5_frame_idx)
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

        st.session_state.page5_frame_idx = frame
        st.session_state.page5_elapsed_in_fixation = new_elapsed

        if ended:
            st.session_state.page5_is_playing = False

        time.sleep(display_step_s)
        st.rerun()