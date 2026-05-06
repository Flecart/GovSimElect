import pandas as pd

import multiprocessing
import pandas as pd
from difflib import SequenceMatcher
from tqdm import tqdm
from functools import partial

df = pd.read_csv("data/contextualization-filtered-fixed.csv")
mit = pd.read_csv("data/mit.csv")

# DEPRECATED: This script is no longer in active use, as the domain assignment
# has been integrated into other parts of the pipeline. However, it is kept here
# to check with anonymous.


# 1. Define the worker function at the top level so it can be pickled
def find_match_index(target_string, descriptions, risks):
    """
    Returns the index of the matching row in 'mit'.
    Prioritizes substring match in 'descriptions', falls back to fuzzy match in 'risks'.
    """
    if pd.isna(target_string):
        return -1

    # Strategy A: Substring Match (Fastest)
    # Check if any description is a substring of the target
    for i, desc in enumerate(descriptions):
        if pd.isna(desc):
            continue
        if desc in target_string:
            return i # Found exact substring match, return immediately

    # Strategy B: Fuzzy Match (Fallback)
    # If no substring match, find the best fuzzy match in risks
    best_idx = -1
    best_ratio = 0.0
    
    for i, risk in enumerate(risks):
        if pd.isna(risk):
            continue
        
        # Optimization: SequenceMatcher is expensive, so we run it only when necessary
        ratio = SequenceMatcher(None, risk, target_string).ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
            
    # Optional: You can set a threshold here (e.g., return -1 if best_ratio < 0.5)
    return best_idx
    
    # Prepare data for multiprocessing
    # Converting to list/tuples is faster for iteration than pandas Series
target_strings = df["taxonomy_leaf"].tolist()
mit_desc = mit["Description"].values
mit_risks = mit["risk"].values

num_processes = multiprocessing.cpu_count()
print(f"Processing {len(target_strings)} items using {num_processes} cores...")

# Create a partial function with fixed arguments (the lookup tables)
# This prevents us from having to pass the 'mit' arrays every single time manually
worker = partial(find_match_index, descriptions=mit_desc, risks=mit_risks)

# 3. Run the Pool
with multiprocessing.Pool(processes=num_processes) as pool:
    # imap allows us to see a progress bar with tqdm
    results = list(tqdm(pool.imap(worker, target_strings), total=len(target_strings)))

# 4. Assign results
df["mit_idx"] = results


print("Done.")

mit["index"] = mit.index
# merge value on df with mit on risk, keep only Domain from mit
df_merged = pd.merge(df, mit[['index', 'Domain']], left_on='mit_idx', right_on='index', how='left')
df_merged.head()

print(df_merged['Domain'].value_counts())