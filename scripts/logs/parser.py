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

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        text = f.read()
    df = parse_summaries(text)
    print(f"Parsed {len(df)} rows")
    print(df.head())
    df.to_csv("/home/claude/govsim/parsed.csv", index=False)