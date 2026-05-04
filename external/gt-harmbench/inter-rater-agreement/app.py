"""
Streamlit UI for tagging game types and tracking inter-rater agreement.

Features
- Select any CSV in the ../data folder.
- Readable display of story_row / story_col.
- Per-row game type selection using templates from game_template.csv.
- Save current inferences to disk.
- Toggle display of computer inter-rater agreement vs provided formal_game.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SAMPLES_PER_GAME = 5
RANDOM_SEED = 42
GAME_TYPES: List[str] = [
    "Prisoner's Dilemma",
    "Chicken",
    "Bach or Stravinski",
    "No conflict",
    "Stag hunt",
    "Coordination",
]
assert SAMPLES_PER_GAME * len(GAME_TYPES) == 30, "Expected 5 * 6 = 30 samples"
TEMPLATE_OPTIONS: List[str] = GAME_TYPES + ["Matching pennies"]


@st.cache_data(show_spinner=False)
def list_csv_files() -> List[Path]:
    return sorted(DATA_DIR.glob("*.csv"))


@st.cache_data(show_spinner=False)
def load_dataset(path: Path, sample: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if sample and "formal_game" in df.columns:
        # Stratified sampling: 5 from each game type (excluding Matching pennies)
        samples = []
        for game_type in GAME_TYPES:
            game_df = df[df["formal_game"] == game_type]
            if len(game_df) >= SAMPLES_PER_GAME:
                samples.append(game_df.sample(n=SAMPLES_PER_GAME, random_state=RANDOM_SEED))
            else:
                samples.append(game_df)
        df = pd.concat(samples, ignore_index=True)
        df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def get_label_state(file_key: str) -> Dict[str, str]:
    if "labels" not in st.session_state:
        st.session_state.labels = {}
    if file_key not in st.session_state.labels:
        st.session_state.labels[file_key] = {}
    return st.session_state.labels[file_key]


def sync_labels_from_editor(
    df_slice: pd.DataFrame, edited: pd.DataFrame, labels: Dict[str, str], row_id_col: str
) -> None:
    for _, row in edited.iterrows():
        row_id = str(row[row_id_col])
        selection = row.get("selected_game", "")
        if selection:
            labels[row_id] = selection
        elif row_id in labels:
            labels.pop(row_id)


def compute_agreement(df: pd.DataFrame, labels: Dict[str, str], row_id_col: str) -> Optional[float]:
    if "formal_game" not in df.columns:
        return None
    aligned = []
    for _, row in df.iterrows():
        row_id = str(row[row_id_col])
        if row_id in labels:
            aligned.append(labels[row_id] == str(row["formal_game"]))
    if not aligned:
        return None
    return sum(aligned) / len(aligned)


def save_inferences(
    df: pd.DataFrame, labels: Dict[str, str], row_id_col: str, target_path: Path
) -> None:
    rows = []
    for _, row in df.iterrows():
        row_id = str(row[row_id_col])
        if row_id in labels:
            rows.append(
                {
                    row_id_col: row_id,
                    "story_row": row.get("story_row", ""),
                    "story_col": row.get("story_col", ""),
                    "selected_game": labels[row_id],
                }
            )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target_path, index=False)


def load_saved_inferences(target_path: Path, row_id_col: str) -> Dict[str, str]:
    if not target_path.exists():
        return {}
    saved = pd.read_csv(target_path, dtype=str, keep_default_na=False)
    if row_id_col not in saved.columns or "selected_game" not in saved.columns:
        return {}
    return dict(zip(saved[row_id_col].astype(str), saved["selected_game"]))


def main() -> None:
    st.set_page_config(page_title="Inter-Rater Agreement", layout="wide")
    st.title("Inter-Rater Agreement")
    st.caption(f"Tag games with templates and track agreement. Default dataset samples {SAMPLES_PER_GAME} per game type (30 total).")

    csv_files = list_csv_files()
    if not csv_files:
        st.error(f"No CSV files found in {DATA_DIR}")
        return

    default_file = DATA_DIR / "gt-harmbench.csv"
    default_index = next((i for i, p in enumerate(csv_files) if p == default_file), 0)

    with st.sidebar:
        st.header("Dataset")
        lock_to_default = not st.toggle("Choose a different CSV", value=False)
        if lock_to_default and default_file.exists():
            selected_file = default_file
            st.write(f"Using default: `{selected_file.name}`")
        else:
            selected_file = st.selectbox(
                "Choose CSV",
                csv_files,
                format_func=lambda p: p.name,
                key="file_select",
                index=default_index,
            )

    # Sample 30 items from the default gt-harmbench.csv
    use_sampling = selected_file == default_file
    df = load_dataset(selected_file, sample=use_sampling)
    if not {"story_row", "story_col"}.issubset(df.columns):
        st.error("Selected CSV must contain 'story_row' and 'story_col' columns.")
        return

    row_id_col = "id" if "id" in df.columns else df.columns[0]
    labels = get_label_state(selected_file.name)
    output_path = OUTPUT_DIR / f"inferences_{selected_file.stem}.csv"
    if not labels:
        labels.update(load_saved_inferences(output_path, row_id_col))

    total_rows = len(df)

    # Navigation: one sample per page
    col_prev, col_nav, col_next = st.columns([1, 2, 1])
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    with col_prev:
        if st.button("← Previous", disabled=st.session_state.current_idx == 0):
            st.session_state.current_idx -= 1
            st.rerun()
    with col_nav:
        st.markdown(f"**Sample {st.session_state.current_idx + 1} of {total_rows}** &nbsp;|&nbsp; Tagged: {len(labels)}/{total_rows}")
    with col_next:
        if st.button("Next →", disabled=st.session_state.current_idx >= total_rows - 1):
            st.session_state.current_idx += 1
            st.rerun()

    # Get current row
    idx = st.session_state.current_idx
    current_row = df.iloc[idx]
    row_id = str(current_row[row_id_col])

    st.divider()

    # Display Row Player story
    st.subheader("Row Player")
    st.markdown(current_row["story_row"])

    st.divider()

    # Display Column Player story
    st.subheader("Column Player")
    st.markdown(current_row["story_col"])

    st.divider()

    # Game type selection
    current_label = labels.get(row_id, "")
    options = [""] + TEMPLATE_OPTIONS
    current_index = options.index(current_label) if current_label in options else 0

    selected_game = st.selectbox(
        "Select game type",
        options,
        index=current_index,
        key=f"game_select_{row_id}",
    )

    # Update label
    if selected_game:
        labels[row_id] = selected_game
    elif row_id in labels:
        labels.pop(row_id)

    # Action buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save inferences", type="primary"):
            save_inferences(df, labels, row_id_col, output_path)
            st.success(f"Saved to {output_path}")

    with col2:
        toggle_label = (
            "Hide inter-rater agreement" if st.session_state.get("show_agreement") else "Show inter-rater agreement"
        )
        if st.button(toggle_label):
            st.session_state.show_agreement = not st.session_state.get("show_agreement", False)

    if st.session_state.get("show_agreement", False):
        agreement = compute_agreement(df, labels, row_id_col)
        if agreement is None:
            st.warning("Agreement unavailable: no 'formal_game' column or no labeled rows.")
        else:
            st.metric("Computer inter-rater agreement", f"{agreement:.0%}")


if __name__ == "__main__":
    main()




