"""
Streamlit UI for rating samples as good or bad and tracking inter-rater agreement.

Features:
- Merges gt-harmbench.csv (good) and gt-harmbench-bad.csv (bad) datasets
- Single sample per page with navigation
- Displays taxonomy path, game type, payoff matrix, and stories
- Human rating: good or bad
- Inter-rater agreement matrix showing agreement percentages
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@st.cache_data(show_spinner=False)
def load_and_merge_datasets() -> pd.DataFrame:
    """Load and merge good and bad datasets, adding a source column."""
    good_path = DATA_DIR / "gt-harmbench.csv"
    bad_path = DATA_DIR / "gt-harmbench-bad.csv"
    
    if not good_path.exists():
        raise FileNotFoundError(f"File not found: {good_path}")
    if not bad_path.exists():
        raise FileNotFoundError(f"File not found: {bad_path}")
    
    df_good = pd.read_csv(good_path, dtype=str, keep_default_na=False)
    df_bad = pd.read_csv(bad_path, dtype=str, keep_default_na=False)
    
    # Add source column
    df_good["source"] = "good"
    df_bad["source"] = "bad"
    
    # Ensure both have an id column for merging
    if "id" not in df_good.columns:
        df_good.insert(0, "id", range(len(df_good)))
    if "id" not in df_bad.columns:
        df_bad.insert(0, "id", range(len(df_bad)))
    
    # Create unique IDs by prefixing with source
    df_good["id"] = "good_" + df_good["id"].astype(str)
    # keep only 50
    df_good = df_good.sample(n=50, random_state=42).reset_index(drop=True)
    df_bad["id"] = "bad_" + df_bad["id"].astype(str)
    # keep only 50
    df_bad = df_bad.sample(n=50, random_state=42).reset_index(drop=True)
    
    # Merge datasets
    df_merged = pd.concat([df_good, df_bad], ignore_index=True)
    #shuffle
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_merged


def parse_payoff_matrix(row: pd.Series) -> Optional[List[List[List[int]]]]:
    """Parse payoff matrix from row data."""
    try:
        payoff_11 = ast.literal_eval(row.get("1_1_payoff", "[0, 0]"))
        payoff_12 = ast.literal_eval(row.get("1_2_payoff", "[0, 0]"))
        payoff_21 = ast.literal_eval(row.get("2_1_payoff", "[0, 0]"))
        payoff_22 = ast.literal_eval(row.get("2_2_payoff", "[0, 0]"))
        
        return [
            [payoff_11, payoff_12],
            [payoff_21, payoff_22]
        ]
    except (ValueError, SyntaxError):
        return None


def parse_actions(row: pd.Series) -> Tuple[List[str], List[str]]:
    """Parse actions from row data."""
    try:
        actions_row = ast.literal_eval(row.get("actions_row", "[]"))
        actions_col = ast.literal_eval(row.get("actions_column", "[]"))
        return actions_row, actions_col
    except (ValueError, SyntaxError):
        return [], []


def format_payoff_matrix(payoff_matrix: List[List[List[int]]], actions_row: List[str], actions_col: List[str]) -> str:
    """Format payoff matrix as a readable table."""
    if not payoff_matrix or not actions_row or not actions_col:
        return "Invalid payoff matrix"
    
    # Create header
    header = f"| | {actions_col[0]} | {actions_col[1]} |\n"
    separator = "| --- | --- | --- |\n"
    
    # Create rows
    row1 = f"| **{actions_row[0]}** | ({payoff_matrix[0][0][0]}, {payoff_matrix[0][0][1]}) | ({payoff_matrix[0][1][0]}, {payoff_matrix[0][1][1]}) |\n"
    row2 = f"| **{actions_row[1]}** | ({payoff_matrix[1][0][0]}, {payoff_matrix[1][0][1]}) | ({payoff_matrix[1][1][0]}, {payoff_matrix[1][1][1]}) |\n"
    
    return header + separator + row1 + row2


def get_taxonomy_path(row: pd.Series) -> str:
    """Build taxonomy path from row data."""
    parts = []
    
    ev_id = row.get("Ev_ID", "").strip()
    if ev_id:
        parts.append(f"Ev_ID: {ev_id}")
    
    risk_category = row.get("Risk category", "").strip()
    if risk_category:
        parts.append(risk_category)
    
    risk_subcategory = row.get("Risk subcategory", "").strip()
    if risk_subcategory:
        parts.append(risk_subcategory)
    
    description = row.get("Description", "").strip()
    if description:
        parts.append(description)
    
    return " > ".join(parts) if parts else "No taxonomy information"


def get_rating_state() -> Dict[str, str]:
    """Get or initialize rating state (good/bad labels)."""
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}
    return st.session_state.ratings


def load_saved_ratings(target_path: Path) -> Dict[str, str]:
    """Load saved ratings from CSV."""
    if not target_path.exists():
        return {}
    try:
        saved = pd.read_csv(target_path, dtype=str, keep_default_na=False)
        if "id" not in saved.columns or "rating" not in saved.columns:
            return {}
        return dict(zip(saved["id"].astype(str), saved["rating"]))
    except Exception:
        return {}


def save_ratings(df: pd.DataFrame, ratings: Dict[str, str], target_path: Path) -> None:
    """Save ratings to CSV."""
    rows = []
    for _, row in df.iterrows():
        row_id = str(row["id"])
        if row_id in ratings:
            rows.append({
                "id": row_id,
                "source": row.get("source", ""),
                "rating": ratings[row_id],
                "Ev_ID": row.get("Ev_ID", ""),
                "formal_game": row.get("formal_game", ""),
            })
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target_path, index=False)


def compute_agreement_matrix(df: pd.DataFrame, ratings: Dict[str, str]) -> Optional[pd.DataFrame]:
    """Compute agreement matrix between source (good/bad) and human ratings."""
    if not ratings:
        return None
    
    # Count agreements
    marked_good_good = 0  # Source: good, Human: good
    marked_good_bad = 0   # Source: good, Human: bad
    marked_bad_good = 0   # Source: bad, Human: good
    marked_bad_bad = 0    # Source: bad, Human: bad
    
    for _, row in df.iterrows():
        row_id = str(row["id"])
        if row_id in ratings:
            source = row.get("source", "")
            human_rating = ratings[row_id]
            
            if source == "good" and human_rating == "good":
                marked_good_good += 1
            elif source == "good" and human_rating == "bad":
                marked_good_bad += 1
            elif source == "bad" and human_rating == "good":
                marked_bad_good += 1
            elif source == "bad" and human_rating == "bad":
                marked_bad_bad += 1
    
    total = marked_good_good + marked_good_bad + marked_bad_good + marked_bad_bad
    if total == 0:
        return None
    
    # Create matrix with percentages
    matrix_data = {
        "": ["Marked good", "Marked bad"],
        "Good": [
            f"{marked_good_good} ({marked_good_good/total*100:.1f}%)" if total > 0 else "0 (0.0%)",
            f"{marked_bad_good} ({marked_bad_good/total*100:.1f}%)" if total > 0 else "0 (0.0%)"
        ],
        "Bad": [
            f"{marked_good_bad} ({marked_good_bad/total*100:.1f}%)" if total > 0 else "0 (0.0%)",
            f"{marked_bad_bad} ({marked_bad_bad/total*100:.1f}%)" if total > 0 else "0 (0.0%)"
        ]
    }
    
    return pd.DataFrame(matrix_data)


def main() -> None:
    st.set_page_config(page_title="Sample Rating", layout="wide")
    st.title("Sample Rating: Good or Bad")
    st.caption("Rate merged samples from gt-harmbench (good) and gt-harmbench-bad (bad) datasets.")
    
    try:
        df = load_and_merge_datasets()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    
    if len(df) == 0:
        st.error("No data loaded.")
        return
    
    ratings = get_rating_state()
    output_path = OUTPUT_DIR / "ratings_merged.csv"
    
    # Load saved ratings if state is empty
    if not ratings:
        ratings.update(load_saved_ratings(output_path))
    
    total_samples = len(df)
    
    # Initialize current index in session state
    if "current_sample_index" not in st.session_state:
        st.session_state.current_sample_index = 0
    
    # Get current index (set before sidebar so it's available everywhere)
    current_index = st.session_state.current_sample_index
    
    # Navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Use number_input to display and allow manual entry
        # Don't use a key to avoid conflicts - just use value prop
        number_input_value = st.number_input(
            "Sample index",
            min_value=0,
            max_value=total_samples - 1,
            value=current_index,
            step=1
        )
        
        # Sync: if user manually changed the number input, update our state
        if number_input_value != st.session_state.current_sample_index:
            st.session_state.current_sample_index = number_input_value
        
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("← Previous", disabled=(current_index == 0)):
                st.session_state.current_sample_index = max(0, current_index - 1)
                st.rerun()
        with col_next:
            if st.button("Next →", disabled=(current_index == total_samples - 1)):
                st.session_state.current_sample_index = min(total_samples - 1, current_index + 1)
                st.rerun()
        
        st.write(f"Sample {current_index + 1} of {total_samples}")
        st.write(f"Rated: {len(ratings)} / {total_samples}")
        
        # Show source
        current_row_sidebar = df.iloc[current_index]
        source = current_row_sidebar.get("source", "unknown")
        source_color = "🟢" if source == "good" else "🔴"
        st.write(f"Source: Redacted")# {source_color} {source}")
    
    # Update current_index after sidebar (in case it changed)
    current_index = st.session_state.current_sample_index
    
    # Display current sample
    current_row = df.iloc[current_index]
    row_id = str(current_row["id"])
    
    # Taxonomy path
    st.subheader("Taxonomy Path")
    taxonomy_path = get_taxonomy_path(current_row)
    st.write(taxonomy_path)
    
    # Game type
    st.subheader("Game Type")
    formal_game = current_row.get("formal_game", "Unknown")
    st.write(f"**{formal_game}**")
    
    # Scenario safety level
    st.subheader("Scenario Safety Level")
    risk_level = current_row.get("risk_level", "Unknown")
    st.write(f"**{risk_level}**")
    
    # Payoff matrix
    st.subheader("Payoff Matrix")
    payoff_matrix = parse_payoff_matrix(current_row)
    actions_row, actions_col = parse_actions(current_row)
    
    if payoff_matrix:
        matrix_markdown = format_payoff_matrix(payoff_matrix, actions_row, actions_col)
        st.markdown(matrix_markdown)
    else:
        st.warning("Could not parse payoff matrix")
    
    # Stories
    st.subheader("Row Story")
    story_row = current_row.get("story_row", "")
    st.text_area("", story_row, height=200, disabled=False, key=f"story_row_{current_index}")
    
    st.subheader("Column Story")
    story_col = current_row.get("story_col", "")
    st.text_area("", story_col, height=200, disabled=False, key=f"story_col_{current_index}")
    
    # Rating section
    st.divider()
    st.subheader("Rating")
    
    current_rating = ratings.get(row_id, "")
    
    col_good, col_bad, col_clear = st.columns([1, 1, 1])
    
    with col_good:
        if st.button("✅ Good", type="primary" if current_rating == "good" else "secondary", use_container_width=True):
            ratings[row_id] = "good"
            st.rerun()
    
    with col_bad:
        if st.button("❌ Bad", type="primary" if current_rating == "bad" else "secondary", use_container_width=True):
            ratings[row_id] = "bad"
            st.rerun()
    
    with col_clear:
        if st.button("Clear", use_container_width=True):
            if row_id in ratings:
                del ratings[row_id]
            st.rerun()
    
    if current_rating:
        st.info(f"Current rating: **{current_rating.upper()}**")
    
    # Action buttons
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("💾 Save Ratings", type="primary", use_container_width=True):
            save_ratings(df, ratings, output_path)
            st.success(f"Saved to {output_path}")
    
    with col2:
        toggle_label = (
            "Hide inter-rater agreement" if st.session_state.get("show_agreement") else "Show inter-rater agreement"
        )
        if st.button(toggle_label, use_container_width=True):
            st.session_state.show_agreement = not st.session_state.get("show_agreement", False)
            st.rerun()
    
    # Inter-rater agreement matrix
    if st.session_state.get("show_agreement", False):
        st.divider()
        st.subheader("Inter-Rater Agreement")
        
        agreement_matrix = compute_agreement_matrix(df, ratings)
        if agreement_matrix is None:
            st.warning("No ratings available to compute agreement.")
        else:
            st.dataframe(agreement_matrix, use_container_width=True, hide_index=True)
            
            # Calculate overall agreement
            if len(ratings) > 0:
                total_agreements = 0
                total_ratings = 0
                for _, row in df.iterrows():
                    row_id_check = str(row["id"])
                    if row_id_check in ratings:
                        total_ratings += 1
                        source = row.get("source", "")
                        human_rating = ratings[row_id_check]
                        if source == human_rating:
                            total_agreements += 1
                
                if total_ratings > 0:
                    overall_agreement = total_agreements / total_ratings
                    st.metric("Overall Agreement", f"{overall_agreement:.1%}")


if __name__ == "__main__":
    main()
