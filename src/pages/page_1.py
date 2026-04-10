from __future__ import annotations

import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_processing.data_processing import get_data_dirs
from src.saliency.static_salience import get_game_metadata, salience_static


AOI_COUNT = 100
GRID_SIZE = 10
TIME_STEP_MS = 9


def aoi_to_matrix(values: np.ndarray) -> np.ndarray:
    """Convert a flat AOI vector (1..100) to a 10x10 matrix (row-wise)."""
    return np.array(values).reshape(GRID_SIZE, GRID_SIZE)


def aoi_to_coordinates(aoi_values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Map AOI ids (1..100) to heatmap coordinates centered in each cell."""
    idx = aoi_values.to_numpy(dtype=int) - 1
    rows = idx // GRID_SIZE
    cols = idx % GRID_SIZE
    return cols + 0.5, rows + 0.5


def is_supported_game_sheet(sheet_name: str) -> bool:
    """Check whether a sheet can be mapped to GAME1..GAME6."""
    try:
        get_game_metadata(sheet_name)
        return True
    except ValueError:
        return False


def normalize_fixations(raw_df: pd.DataFrame) -> pd.DataFrame:
    data = raw_df.copy()

    # Feuille vide ou sans colonnes
    if raw_df is None or raw_df.empty or raw_df.shape[1] == 0:
        return pd.DataFrame(columns=["AOI", "Time"])

    # Si moins de 2 colonnes, on ignore proprement
    if data.shape[1] < 2:
        return pd.DataFrame(columns=["AOI", "Time"])

    # On garde uniquement les 2 premières colonnes
    data = data.iloc[:, :2].copy()
    data.columns = ["AOI", "Time"]

    data = data.dropna(subset=["AOI", "Time"]).reset_index(drop=True)

    if data.empty:
        return pd.DataFrame(columns=["AOI", "Time"])

    data["AOI"] = pd.to_numeric(data["AOI"], errors="coerce")
    data["Time"] = pd.to_numeric(data["Time"], errors="coerce")

    data = data.dropna(subset=["AOI", "Time"]).reset_index(drop=True)

    if data.empty:
        return pd.DataFrame(columns=["AOI", "Time"])

    data["AOI"] = data["AOI"].astype(int)
    data["Time"] = data["Time"].astype(float)

    return data


@st.cache_data(show_spinner=False)
def discover_player_files(project_root: str) -> dict[str, str]:
    """Discover Excel player files in project tree and configured data folders."""
    root = Path(project_root)
    data_dir, output_dir = get_data_dirs()
    blocked_parts = {".git", ".idea", "__pycache__", ".venv", "venv"}
    found: list[Path] = []

    for pattern in ("*.xlsx", "*.xls", "*.xlsm"):
        for path in root.rglob(pattern):
            if not blocked_parts.intersection(path.parts):
                found.append(path)

    for extra_dir in (output_dir, data_dir):
        if extra_dir.exists():
            for pattern in ("*.xlsx", "*.xls", "*.xlsm"):
                found.extend(extra_dir.glob(pattern))

    # Prefer already processed workbooks when both raw/processed versions exist.
    files: dict[str, str] = {}
    unique_paths = sorted(set(found), key=lambda p: ("_AOI_TIME" not in p.stem.upper(), str(p)))
    for path in unique_paths:
        if path.name.startswith("~$"):
            continue
        label = path.stem
        if label in files:
            label = f"{label} ({path.parent.name})"
        files[label] = str(path)
    return files


@st.cache_data(show_spinner=False)
def load_workbook_sheets(workbook_path: str) -> dict[str, pd.DataFrame]:
    """Load and clean all sheets from one player workbook."""
    excel_file = pd.ExcelFile(workbook_path)
    sheets: dict[str, pd.DataFrame] = {}
    for sheet_name in excel_file.sheet_names:
        raw_df = pd.read_excel(excel_file, sheet_name=sheet_name)
        sheets[sheet_name] = normalize_fixations(raw_df)
    return sheets


def compute_static_score(fixations: pd.DataFrame, q: np.ndarray, upto: int | None = None) -> float:
    """Compute normalized time-weighted static salience score in [0, 100]."""
    if fixations.empty:
        return 0.0

    data = fixations if upto is None else fixations.iloc[:upto]
    if data.empty:
        return 0.0

    weights = data["Time"].to_numpy(dtype=float)
    q_values = q[data["AOI"].to_numpy(dtype=int) - 1]
    weighted_mean = float(np.sum(q_values * weights) / np.sum(weights))
    return 100.0 * weighted_mean / float(np.max(q))


def extract_game_number(sheet_name: str) -> int:
    """Extract numeric game id from a sheet name like GAME4."""
    match = re.search(r"(\d+)", str(sheet_name).upper())
    if not match:
        raise ValueError(f"Cannot extract game number from '{sheet_name}'.")
    return int(match.group(1))


def show_page_1():
    st.header("Cartes de saillance statiques")
    st.caption(
        "Score statique = moyenne ponderee par le temps des probabilites de saillance q(AOI), "
        "normalisee sur [0, 100]."
    )

    project_root = Path(__file__).resolve().parents[2]
    data_dir, output_dir = get_data_dirs()
    player_files = discover_player_files(str(project_root))

    if not player_files:
        st.warning("Aucun fichier Excel joueur trouve.")
        st.info(
            f"Dossiers scannes:\n- {project_root}\n- {output_dir}\n- {data_dir}\n\n"
            "Place les fichiers Excel joueurs dans l'un de ces dossiers."
        )
        return

    st.caption(
        f"Recherche des fichiers dans: `{project_root}`, `{output_dir}`, `{data_dir}`"
    )

    left_col, center_col, right_col = st.columns([1.1, 2.3, 1.2])

    with left_col:
        st.subheader("Selection")

        player_name = st.selectbox("Joueur", options=list(player_files.keys()))
        workbook_path = player_files[player_name]

        workbook = load_workbook_sheets(workbook_path)
        sheet_names = [name for name in workbook if is_supported_game_sheet(name)]

        if not sheet_names:
            st.warning("Aucune feuille GAME1..GAME6 exploitable dans ce fichier.")
            return

        sheet_names = sorted(sheet_names, key=extract_game_number)
        sheet_name = st.selectbox("Jeu (sheet)", options=sheet_names)

        beta = st.slider("Sensibilite beta (softmax)", 0.0001, 0.5, 0.01, 0.0001, format="%.4f")
        speed = st.slider("Facteur de vitesse lecture", 0.1, 20.0, 1.0, 0.1)

        with st.expander("Parametres de saillance (λ)"):
            st.caption(
                "λ1 : poids du point fixe · "
                "λ2 : poids des multiples de 3 · "
                "λ3 : poids des reperes ronds {15,30,45,60,75,90} · "
                "λ_blur : diffusion spatiale"
            )
            normalize_components = st.checkbox(
                "Normaliser les composantes avant combinaison (recommande)",
                value=True,
                help="Normaliser chaque composante sur [0, 1] avant de les combiner avec les poids λ rend les valeurs plus comparables et interpretable. Sans normalisation, les composantes avec des echelles plus grandes peuvent dominer le score final.",
            )
            lambda1 = st.slider("λ1 — point fixe S_fp", 0.0, 3.0, 1.0, 0.05)
            lambda2 = st.slider("λ2 — multiples de 3 S_mod", 0.0, 3.0, 0.5, 0.05)
            lambda3 = st.slider("λ3 — reperes ronds S_round", 0.0, 3.0, 0.3, 0.05)
            lambda_blur = st.slider("λ_blur — diffusion spatiale S_blur", 0.0, 3.0, 0.4, 0.05)
            blur_sigma = st.slider("Sigma blur spatial", 0.3, 4.0, 1.1, 0.05)

        # Component to display on the heatmap (score always uses q)
        COMPONENT_LABELS: dict[str, str] = {
            "q - distribution finale": "q",
            "S - saillance totale": "S",
            "S_fp_norm - point fixe normalise": "S_fp_norm",
            "S_mod_norm - multiples de 3 normalise": "S_mod_norm",
            "S_round_norm - reperes ronds normalise": "S_round_norm",
            "S_blur_norm - diffusion normalisee": "S_blur_norm",
            "S_fp - point fixe brut": "S_fp",
            "S_mod - multiples de 3 brut": "S_mod",
            "S_round - reperes ronds brut": "S_round",
            "S_blur - diffusion brute": "S_blur",
        }
        display_label = st.selectbox(
            "Composante a visualiser",
            options=list(COMPONENT_LABELS.keys()),
            index=0,
        )
        display_key = COMPONENT_LABELS[display_label]

    fixations = workbook[sheet_name]
    game_meta = get_game_metadata(sheet_name)
    salience_result = salience_static(
        sheet_name,
        beta=beta,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        lambda_blur=lambda_blur,
        blur_sigma=blur_sigma,
        normalize_components=normalize_components,
    )
    q = np.asarray(salience_result["q"], dtype=float)

    # The display matrix follows the user's component choice.
    # For binary / raw components we normalise to [0, 1] so the colormap is consistent.
    display_values = np.asarray(salience_result[display_key], dtype=float)
    v_min, v_max = display_values.min(), display_values.max()
    display_normalised = (display_values - v_min) / (v_max - v_min + 1e-12)
    matrix = aoi_to_matrix(display_normalised)


    selection_key = f"{workbook_path}::{sheet_name}"
    if st.session_state.get("selection_key") != selection_key:
        st.session_state.selection_key = selection_key
        st.session_state.frame_idx = 0
        st.session_state.is_playing = False
        st.session_state.elapsed_in_fixation = 0
        st.session_state.last_rerun_time = None

    st.session_state.setdefault("frame_idx", 0)
    st.session_state.setdefault("is_playing", False)
    st.session_state.setdefault("elapsed_in_fixation", 0)
    st.session_state.setdefault("last_rerun_time", None)

    if fixations.empty:
        st.warning("La feuille selectionnee ne contient pas de fixations valides (AOI, Time).")
        return

    max_idx = len(fixations) - 1
    st.session_state.frame_idx = min(st.session_state.frame_idx, max_idx)

    with left_col:
        play_col, reset_col = st.columns(2)
        if play_col.button("Play", use_container_width=True):
            st.session_state.is_playing = True
            st.session_state.last_rerun_time = time.time()  # démarre le chrono mural
        if reset_col.button("Reset", use_container_width=True):
            st.session_state.frame_idx = 0
            st.session_state.is_playing = False
            st.session_state.elapsed_in_fixation = 0
            st.session_state.last_rerun_time = None

        st.caption(f"Fixation affichee: {st.session_state.frame_idx + 1}/{len(fixations)}")

    visible_fixations = fixations.iloc[: st.session_state.frame_idx + 1]
    x_coords, y_coords = aoi_to_coordinates(visible_fixations["AOI"])

    # Temps de la fixation courante et progression interne
    current_fix_duration_ms = float(fixations.iloc[st.session_state.frame_idx]["Time"])
    elapsed_in_fix_ms = float(st.session_state.elapsed_in_fixation)
    completed_time_ms = float(fixations.iloc[: st.session_state.frame_idx]["Time"].sum())
    cumulative_time_ms = completed_time_ms + elapsed_in_fix_ms
    total_time_ms = float(fixations["Time"].sum())
    current_aoi = int(visible_fixations.iloc[-1]["AOI"])
    fix_count = int(len(visible_fixations))
    current_score = compute_static_score(fixations, q, upto=fix_count)
    final_score = compute_static_score(fixations, q)

    # Contributions pondérées (pour la ligne d'info et right_col)
    c_fp = lambda1 * np.asarray(
        salience_result["S_fp_norm"] if normalize_components else salience_result["S_fp"],
        dtype=float,
    )
    c_mod = lambda2 * np.asarray(
        salience_result["S_mod_norm"] if normalize_components else salience_result["S_mod"],
        dtype=float,
    )
    c_round = lambda3 * np.asarray(
        salience_result["S_round_norm"] if normalize_components else salience_result["S_round"],
        dtype=float,
    )
    c_blur = lambda_blur * np.asarray(
        salience_result["S_blur_norm"] if normalize_components else salience_result["S_blur"],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    sns.heatmap(matrix, cmap="viridis", square=True, cbar=True, ax=ax)

    if len(visible_fixations) > 0:
        ax.plot(x_coords, y_coords, color="white", linewidth=1.2, alpha=0.95)
        ax.scatter(
            x_coords[-1],
            y_coords[-1],
            color="red",
            s=70,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, 0)
    # Center ticks on cells: X labels 1..10, Y labels 0..9
    ax.set_xticks(np.arange(GRID_SIZE) + 0.5)
    ax.set_xticklabels(np.arange(1, GRID_SIZE + 1))
    ax.set_yticks(np.arange(GRID_SIZE) + 0.5)
    ax.set_yticklabels(np.arange(0, GRID_SIZE))
    component_name = display_label.split(" - ")[0].strip()
    ax.set_title(f"{sheet_name} - {game_meta['game_type']} | {component_name}")
    ax.set_xlabel("Colonnes AOI")
    ax.set_ylabel("Lignes AOI")

    with center_col:
        # --- Ligne d'infos horizontale au-dessus de la heatmap ---
        st.markdown(
            f"**Jeu :** `{sheet_name}` &nbsp;·&nbsp; "
            f"**Type :** `{game_meta['game_type']}` &nbsp;·&nbsp; "
            f"**Règle :** `{game_meta['rule']}` &nbsp;·&nbsp; "
            f"**Beta :** `{beta:.4f}` &nbsp;·&nbsp; "
            f"**Sigma :** `{blur_sigma:.2f}` &nbsp;·&nbsp; "
            f"**Norm :** `{salience_result['normalize_components']}`"
        )
        st.markdown(
            f"**λ fp :** `{float(np.mean(c_fp)):.3f}` &nbsp;·&nbsp; "
            f"**λ mod :** `{float(np.mean(c_mod)):.3f}` &nbsp;·&nbsp; "
            f"**λ round :** `{float(np.mean(c_round)):.3f}` &nbsp;·&nbsp; "
            f"**λ blur :** `{float(np.mean(c_blur)):.3f}`"
        )
        # ----------------------------------------------------------
        st.subheader(f"Heatmap ({component_name}) + trajectoire")
        st.pyplot(fig)
        plt.close(fig)

        st.caption("Matrice AOI 10x10 (1 en haut gauche, 100 en bas droite)")
        st.dataframe(aoi_to_matrix(np.arange(1, AOI_COUNT + 1)), use_container_width=True, height=220)

    with right_col:
        st.subheader("Indicateurs")
        st.metric("Score courant", f"{current_score:.2f}/100")
        st.metric("Score final (jeu)", f"{final_score:.2f}/100")
        st.metric("AOI actuelle", str(current_aoi))
        st.metric("Fixations", f"{fix_count}/{len(fixations)}")

        # --- Progression temporelle continue ---
        st.markdown("**Fixation en cours :**")
        fix_progress = min(elapsed_in_fix_ms / max(current_fix_duration_ms, 1.0), 1.0)
        st.caption(f"Temps fixation : `{elapsed_in_fix_ms:.0f} / {current_fix_duration_ms:.0f} ms`")
        st.progress(fix_progress)

        st.markdown("**Progression globale :**")
        global_progress = min(cumulative_time_ms / max(total_time_ms, 1.0), 1.0)
        st.caption(f"Temps cumulé : `{cumulative_time_ms:.0f} / {total_time_ms:.0f} ms`")
        st.progress(global_progress)
        # ----------------------------------------

        st.metric("Temps total", f"{total_time_ms:.0f} ms")

    st.markdown("**AOI parcourues (ordre):**")
    ordered_aoi = visible_fixations["AOI"].astype(str).tolist()
    if len(ordered_aoi) > 40:
        st.code(" -> ".join(ordered_aoi[:40]) + " -> ...")
    else:
        st.code(" -> ".join(ordered_aoi))

    with st.expander("Voir le tableau des fixations"):
        st.dataframe(fixations, use_container_width=True, height=240)

    if st.session_state.is_playing:
        DISPLAY_STEP_S = 0.04  # ~25 fps de rafraîchissement visuel

        now = time.time()
        last_t = st.session_state.last_rerun_time
        # Temps réel écoulé depuis le dernier rerun (cappé à 500 ms pour éviter les sauts)
        real_delta_ms = min((now - last_t) * 1000.0, 500.0) if last_t is not None else 0.0
        st.session_state.last_rerun_time = now

        # Avancement simulé = temps réel × facteur de vitesse
        sim_delta_ms = real_delta_ms * float(speed)
        new_elapsed = st.session_state.elapsed_in_fixation + sim_delta_ms
        frame = st.session_state.frame_idx
        ended = False

        # Avancer à travers autant de fixations que nécessaire
        while True:
            fix_dur = max(float(fixations.iloc[frame]["Time"]), 1.0)
            if new_elapsed < fix_dur:
                break  # toujours dans la fixation courante
            new_elapsed -= fix_dur
            if frame < max_idx:
                frame += 1
            else:
                new_elapsed = fix_dur  # caper à la fin
                ended = True
                break

        st.session_state.frame_idx = frame
        st.session_state.elapsed_in_fixation = new_elapsed
        if ended:
            st.session_state.is_playing = False

        time.sleep(DISPLAY_STEP_S)
        st.rerun()

