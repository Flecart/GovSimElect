# Inter-Rater Agreement Webapp

## Goal
Lightweight UI to speed up rating: pick a dataset, read each `story_row`/`story_col`, label the game type, and watch agreement against the dataset’s `formal_game` when present. Prior saved inferences load automatically so work is not lost.

## Quick start
1) From repo root, install deps (uv recommended):
   - `uv pip install -r inter-rater-agreement/requirements.txt`
2) Run the app:
   - `uv run streamlit run inter-rater-agreement/app.py`
3) In the sidebar:
   - By default it loads `gt-harmbench.csv`; toggle “Choose a different CSV” to pick another file from `data/`.
4) In the table:
   - Select a game template for each row using the dropdown (7 predefined games).
   - Click “Save inferences” to write `outputs/inferences_<csvname>.csv`.
   - Toggle “Show/Hide inter-rater agreement” to view match rate vs `formal_game` when available.

That’s it—label, save, and continue later with saved inferences restored automatically.

