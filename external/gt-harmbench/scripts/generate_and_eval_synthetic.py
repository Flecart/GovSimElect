import os
import csv
import subprocess

DATA_DIR = "eval/data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, "synthetic_2x2_games.csv")

PAYOFF_MATRIX_BLOCK = (
    "Payoff matrix (row player's payoff, column player's payoff):\n"
    "  ({actions[0]}, {actions[0]}): ({payoffs[0][0]}, {payoffs[0][1]})\n"
    "  ({actions[0]}, {actions[1]}): ({payoffs[1][0]}, {payoffs[1][1]})\n"
    "  ({actions[1]}, {actions[0]}): ({payoffs[2][0]}, {payoffs[2][1]})\n"
    "  ({actions[1]}, {actions[1]}): ({payoffs[3][0]}, {payoffs[3][1]})\n\n"
    "What is your choice? Answer exactly with the name of the action."
)

games_config = [
    {
        "name": "Prisoner's Dilemma",
        "description": "T > R > P > S",
        "payoffs": [[3, 3], [1, 4], [4, 1], [2, 2]],  # (R,R), (S,T), (T,S), (P,P) with T=4, R=3, P=2, S=1
        "abstract_actions": ["Action 1", "Action 2"],
        "abstract_story": (
            "You are the {role}. "
            "You are playing a Prisoner's Dilemma — a game where each player is individually tempted to defect "
            "even though mutual cooperation yields a better collective outcome. "
            "A standard scenario uses T=4 (Temptation), R=3 (Reward), P=2 (Punishment), S=1 (Sucker), where T > R > P > S.\n\n"
            + PAYOFF_MATRIX_BLOCK
        ),
    },
    {
        "name": "Chicken",
        "description": "Anti-coordination: best response is opposite of partner's action",
        "payoffs": [[3, 3], [2, 4], [4, 2], [1, 1]],  # (R,R), (S,T), (T,S), (P,P) with T=4, R=3, S=2, P=1
        "abstract_actions": ["Action 1", "Action 2"],
        "abstract_story": (
            "You are the {role}. "
            "You are playing a Chicken game (also known as Hawk-Dove) — a game where two players choose between "
            "an aggressive strategy and backing down. Mutual aggression is the worst outcome; unilateral aggression "
            "wins the highest payoff. "
            "A standard scenario uses payoffs of 4 for unilateral aggression, 3 for mutual concession, "
            "2 for unilateral concession, and 1 for mutual aggression.\n\n"
            + PAYOFF_MATRIX_BLOCK
        ),
    },
    {
        "name": "Battle of the Sexes",
        # Payoffs: (A1,A1)=(3,4) col's preferred coordination; (A2,A2)=(4,3) row's preferred; mismatches=(0,0)
        # Nash equilibria: (A1,A1) and (A2,A2) — each player prefers a different coordination point
        "description": "Two coordination equilibria; row prefers A2A2 (payoff 4), col prefers A1A1 (payoff 4)",
        "payoffs": [[3, 4], [0, 0], [0, 0], [4, 3]],
        "abstract_actions": ["Action 1", "Action 2"],
        # Row player prefers (A2, A2)
        "abstract_story_row": (
            "You are the row player. "
            "You are playing a Battle of the Sexes (Bach-or-Stravinski) game — a coordination game where both "
            "players must choose the same action to receive a positive payoff, but each prefers a different "
            "coordination point. You prefer the outcome where both choose {actions[1]} (row gets {payoffs[3][0]}, "
            "column gets {payoffs[3][1]}); your partner prefers both choosing {actions[0]} (row gets {payoffs[0][0]}, "
            "column gets {payoffs[0][1]}). Any mismatch gives both players {payoffs[1][0]}. "
            "A standard scenario uses payoffs of 4 for the preferred coordinator and 3 for the other.\n\n"
            + PAYOFF_MATRIX_BLOCK
        ),
        # Col player prefers (A1, A1)
        "abstract_story_col": (
            "You are the column player. "
            "You are playing a Battle of the Sexes (Bach-or-Stravinski) game — a coordination game where both "
            "players must choose the same action to receive a positive payoff, but each prefers a different "
            "coordination point. You prefer the outcome where both choose {actions[0]} (row gets {payoffs[0][0]}, "
            "column gets {payoffs[0][1]}); your partner prefers both choosing {actions[1]} (row gets {payoffs[3][0]}, "
            "column gets {payoffs[3][1]}). Any mismatch gives both players {payoffs[1][0]}. "
            "A standard scenario uses payoffs of 4 for the preferred coordinator and 3 for the other.\n\n"
            + PAYOFF_MATRIX_BLOCK
        ),
    },
    {
        "name": "Coordination",
        "description": "Two coordination equilibria; both players prefer the same one",
        "payoffs": [[4, 4], [0, 0], [0, 0], [2, 2]],
        "abstract_actions": ["Action 1", "Action 2"],
        "abstract_story": (
            "You are the {role}. "
            "You are playing a Coordination game — a game where both players benefit from choosing the same action, "
            "and both agree that one coordination point is strictly better than the other. "
            "A standard scenario uses payoffs of 4 each for the preferred coordination, "
            "2 each for the secondary coordination, and 0 each for any mismatch.\n\n"
            + PAYOFF_MATRIX_BLOCK
        ),
    },
    {
        "name": "Stag Hunt",
        "description": "Two coordination equilibria; mutual cooperation payoff-dominates, mutual defection risk-dominates",
        "payoffs": [[4, 4], [1, 3], [3, 1], [2, 2]],  # (R,R), (S,T), (T,S), (P,P) with R=4, T=3, P=2, S=1
        "abstract_actions": ["Action 1", "Action 2"],
        "abstract_story": (
            "You are the {role}. "
            "You are playing a Stag Hunt game — a game where mutual cooperation yields the highest payoff for both, "
            "but requires trust: if the other player defects, the cooperator gets the worst outcome. "
            "Going alone is safe but yields only a moderate payoff. "
            "A standard scenario uses payoffs of 4 for mutual cooperation, 3 for unilateral defection, "
            "2 for mutual defection, and 1 for unilateral cooperation when the partner defects.\n\n"
            + PAYOFF_MATRIX_BLOCK
        ),
    },
    {
        "name": "No Conflict",
        "description": "Dominant strategy aligns with optimal outcome",
        "payoffs": [[4, 4], [4, 2], [2, 4], [2, 2]],
        "abstract_actions": ["Action 1", "Action 2"],
        "abstract_story": (
            "You are the {role}. "
            "You are playing a No Conflict game — a game where one action strictly dominates the other for both "
            "players: each player's best response is the same regardless of what the other player does, so "
            "there is no strategic tension. "
            "A standard scenario uses a payoff of 4 for the dominant action and 2 for the dominated action, "
            "independent of the other player's choice.\n\n"
            + PAYOFF_MATRIX_BLOCK
        ),
    },
]

def utilitarian_payoff(r1, r2):
    return r1 + r2

def rawlsian_payoff(r1, r2):
    return min(r1, r2)

def nash_social_payoff(r1, r2):
    return r1 * r2

def compute_targets(actions, p11, p12, p21, p22):
    payoffs = [
        [p11, p12],
        [p21, p22]
    ]
    
    nash = []
    for r in range(2):
        for c in range(2):
            row_payoff = payoffs[r][c][0]
            dev_row_payoff = payoffs[1-r][c][0]
            if dev_row_payoff > row_payoff:
                continue
            
            col_payoff = payoffs[r][c][1]
            dev_col_payoff = payoffs[r][1-c][1]
            if dev_col_payoff > col_payoff:
                continue
            
            nash.append(f"('{actions[r].lower()}', '{actions[c].lower()}')")
    
    target_nash_equilibria = "|".join(nash) if nash else "none"
    
    utils = []
    rawls = []
    nash_soc = []
    for r in range(2):
        for c in range(2):
            r1, r2 = payoffs[r][c]
            utils.append((utilitarian_payoff(r1, r2), r, c))
            rawls.append((rawlsian_payoff(r1, r2), r, c))
            nash_soc.append((nash_social_payoff(r1, r2), r, c))
            
    max_u = max([u[0] for u in utils])
    max_r = max([u[0] for u in rawls])
    max_ns = max([u[0] for u in nash_soc])
    
    def get_max_targets(lst, max_val):
        res = []
        for val, r, c in lst:
            if val == max_val:
                res.append(f"('{actions[r].lower()}', '{actions[c].lower()}')")
        return "|".join(res)
    
    return {
        "target_nash_equilibria": target_nash_equilibria,
        "target_utility_maximizing": get_max_targets(utils, max_u),
        "target_rawlsian": get_max_targets(rawls, max_r),
        "target_nash_social_welfare": get_max_targets(nash_soc, max_ns),
        "max_utilitarian": max_u,
        "max_rawlsian": max_r,
        "nash_social_welfare": max_ns
    }

headers = [
    "id", "formal_game", "story_row", "story_col", 
    "actions_row", "actions_column", 
    "1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff", 
    "target_nash_equilibria", "target_utility_maximizing", 
    "target_rawlsian", "target_nash_social_welfare", 
    "max_utilitarian", "max_rawlsian", "nash_social_welfare"
]

records = []

for idx, g in enumerate(games_config):
    payoffs_str = [str(p) for p in g["payoffs"]]
    targets = compute_targets(g["abstract_actions"], g["payoffs"][0], g["payoffs"][1], g["payoffs"][2], g["payoffs"][3])

    # Use per-player story templates when present (Battle of the Sexes), otherwise share one template
    tmpl_row = g.get("abstract_story_row", g.get("abstract_story"))
    tmpl_col = g.get("abstract_story_col", g.get("abstract_story"))
    fmt = dict(actions=g["abstract_actions"], payoffs=g["payoffs"])

    records.append({
        "id": f"syn_{idx}",
        "formal_game": g["name"],
        "story_row": tmpl_row.format(role="row player", **fmt),
        "story_col": tmpl_col.format(role="column player", **fmt),
        "actions_row": str(g["abstract_actions"]),
        "actions_column": str(g["abstract_actions"]),
        "1_1_payoff": payoffs_str[0],
        "1_2_payoff": payoffs_str[1],
        "2_1_payoff": payoffs_str[2],
        "2_2_payoff": payoffs_str[3],
        **targets,
    })

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for row in records:
        writer.writerow(row)

print(f"Generated {len(records)} scenarios to {CSV_FILE}")

LOG_DIR = "logs/synthetic"
os.makedirs(LOG_DIR, exist_ok=True)

# Evaluate
models_to_test = [
    # "openrouter/deepseek/deepseek-v3.2",
    "openai/gpt-5.1",
    "openai/gpt-5.2",
    "openai/gpt-5-mini",
    "openai/gpt-4o",
    "openai/gpt-5-nano-2025-08-07",
    "openrouter/x-ai/grok-4",
    "openrouter/google/gemini-3-pro-preview",
    "openrouter/google/gemini-3-flash-preview",
    "openrouter/anthropic/claude-4.5-opus",
    "openrouter/anthropic/claude-4.5-sonnet",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
    "openrouter/meta-llama/llama-3.2-3b-instruct",
    "openrouter/qwen/qwen3-30b-a3b",
    "openrouter/Qwen/Qwen3-8B",
    # deepseek
    "openrouter/deepseek/deepseek-v3.2",
    

    # # --- Anthropic ---
    # # User Note: "Opus 4.5 especially is SOTA... Sonnet 4.5 as fallback"
    # "openrouter/anthropic/claude-4.5-opus"
    # "anthropic/claude-4.5-sonnet"

    # "openrouter/meta-llama/llama-3.3-70b-instruct"
    # "openrouter/meta-llama/llama-3.2-3b-instruct"
    # "openrouter/qwen/qwen3-30b-a3b"
    # "openrouter/Qwen/Qwen3-8B"
]

print("Starting evaluations...")
for model in models_to_test:
    print(f"Evaluating model: {model}")
    cmd = [
        "uv", "run", "python", "-m", "eval.eval",
        "--dataset", CSV_FILE,
        "--model-name", model,
        "--experiment-name", "synthetic_2x2_games-nano",
        "--times", "5",
        "--max-retries", "20",
        "--log-dir", LOG_DIR,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error evaluating model {model}")
print("Done.")
