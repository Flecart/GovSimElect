"""Parse GovSim summary blocks into a DataFrame."""
import re
import pandas as pd
import sys

def parse_summaries(text: str) -> pd.DataFrame:
    # Split on "--- Summary ---"
    blocks = [b.strip() for b in text.split("--- Summary ---") if b.strip()]
    rows = []
    for b in blocks:
        row = {}
        # Directory
        m = re.search(r"Directory:\s*(\S+)", b)
        if not m:
            continue
        path = m.group(1)
        # Extract model, prosocial, condition
        # pattern: .../<model>-p<k>-<timestamp>/<cond_id>-<cond_name>-p<k>
        # e.g. gpt-4o-p0-2026-04-15-18:04/0-no-contract-p0
        base = path.split("/")[-2] + "/" + path.split("/")[-1]
        # parent folder e.g. gpt-4o-p0-2026-04-15-18:04
        parent = path.split("/")[-2]
        leaf = path.split("/")[-1]
        # model = everything before "-p<digit>-2026"
        m2 = re.match(r"(.+?)-p(\d+)-\d{4}-\d{2}-\d{2}-\d{2}:\d{2}", parent)
        if not m2:
            continue
        model = m2.group(1)
        prosocial = int(m2.group(2))
        # leaf: <id>-<cond>-p<k>
        m3 = re.match(r"(\d+)-(.+?)-p\d+$", leaf)
        if not m3:
            continue
        cond_id = int(m3.group(1))
        cond = m3.group(2)
        row["model"] = model
        row["prosocial"] = prosocial
        row["cond_id"] = cond_id
        row["condition"] = cond
        row["parent_run"] = parent  # used to dedupe

        def grab(pattern):
            mm = re.search(pattern, b)
            if not mm:
                return None, None, None
            return float(mm.group(1)), float(mm.group(2)), float(mm.group(3))

        m_val, m_lo, m_hi = grab(r"Mean survival months m:\s*([\d.]+)\s*\(95% bootstrap CI for mean:\s*\[([\d.]+),\s*([\d.]+)\]\)")
        q_val, q_lo, q_hi = grab(r"Pooled survival rate q:\s*([\d.]+)\s*\(95% bootstrap CI, resample runs:\s*\[([\d.]+),\s*([\d.]+)\]\)")
        R_val, R_lo, R_hi = grab(r"Total gain R:\s*([\d.]+)\s*\(95% bootstrap CI for mean:\s*\[([\d.]+),\s*([\d.]+)\]\)")
        u_val, u_lo, u_hi = grab(r"Efficiency u:\s*([\d.]+)\s*\(95% bootstrap CI for mean:\s*\[([\d.]+),\s*([\d.]+)\]\)")
        e_val, e_lo, e_hi = grab(r"Equality e:\s*([\d.]+)\s*\(95% bootstrap CI for mean:\s*\[([\d.]+),\s*([\d.]+)\]\)")
        o_val, o_lo, o_hi = grab(r"Over-usage o:\s*([\d.]+)\s*\(95% bootstrap CI for mean:\s*\[([\d.]+),\s*([\d.]+)\]\)")

        # Survival decomposition
        ms = re.search(r"full horizon \(n ≥ max_rounds\)=(\d+),\s*below horizon=(\d+)", b)
        full_h, below_h = (int(ms.group(1)), int(ms.group(2))) if ms else (None, None)

        row.update({
            "m": m_val, "m_lo": m_lo, "m_hi": m_hi,
            "q": q_val, "q_lo": q_lo, "q_hi": q_hi,
            "R": R_val, "R_lo": R_lo, "R_hi": R_hi,
            "u": u_val, "u_lo": u_lo, "u_hi": u_hi,
            "e": e_val, "e_lo": e_lo, "e_hi": e_hi,
            "o": o_val, "o_lo": o_lo, "o_hi": o_hi,
            "full_horizon": full_h,
            "below_horizon": below_h,
        })
        rows.append(row)
    return pd.DataFrame(rows)


 
# For models with multiple parent_runs per cell (gpt-5.4-mini), aggregate across runs:
# average the point estimate and average the CI half-widths (they are comparable-sized bootstrap CIs).
def ci_half(lo, hi):
    return (hi - lo) / 2.0
 


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        text = f.read()
        
    if len(sys.argv) > 2:
        end_file = sys.argv[2]
    else:
        end_file = "aggregated.csv"
    df = parse_summaries(text)
    
    
    metrics = ['m', 'q', 'R', 'u', 'e', 'o']
    for mt in metrics:
        df[f'{mt}_hw'] = ci_half(df[f'{mt}_lo'], df[f'{mt}_hi'])
    
    # Determine the full_horizon count - if a cell has 2 parent_runs, they had 5 runs each so total 10;
    # we'll sum full_horizon and below_horizon across parent_runs.
    agg = df.groupby(['model', 'prosocial', 'cond_id', 'condition']).agg(
        m=('m', 'mean'),
        q=('q', 'mean'),
        R=('R', 'mean'),
        u=('u', 'mean'),
        e=('e', 'mean'),
        o=('o', 'mean'),
        m_lo=('m_lo', 'mean'),
        m_hi=('m_hi', 'mean'),
        q_lo=('q_lo', 'mean'),
        q_hi=('q_hi', 'mean'),
        R_lo=('R_lo', 'mean'),
        R_hi=('R_hi', 'mean'),
        u_lo=('u_lo', 'mean'),
        u_hi=('u_hi', 'mean'),
        e_lo=('e_lo', 'mean'),
        e_hi=('e_hi', 'mean'),
        m_hw=('m_hw', 'mean'),
        q_hw=('q_hw', 'mean'),
        R_hw=('R_hw', 'mean'),
        u_hw=('u_hw', 'mean'),
        e_hw=('e_hw', 'mean'),
        o_hw=('o_hw', 'mean'),
        full_horizon=('full_horizon', 'sum'),
        below_horizon=('below_horizon', 'sum'),
        n_parent_runs=('m', 'size'),
    ).reset_index()
    
    agg['total_runs'] = (agg['full_horizon'] + agg['below_horizon']).astype(int)
    print(agg[['model', 'prosocial', 'condition', 'm', 'q', 'R', 'full_horizon', 'total_runs']].to_string())
    agg.to_csv(end_file, index=False)