import pandas as pd

data_2 = pd.read_csv("inter-rater-agreement/outputs/ratings_merged_2.csv")
data_1 = pd.read_csv("inter-rater-agreement/outputs/ratings_merged.csv")

# filter Matching pennies
data_2_mp = data_2[data_2["formal_game"] != "Matching pennies"]
data_1_mp = data_1[data_1["formal_game"] != "Matching pennies"]


def compute_kappa(da, db):
    observed_agreement = (da == db).mean()
    prob_expected = 1/6 # 6 possible ratings
    kappa = (observed_agreement - prob_expected) / (1 - prob_expected)
    return kappa

kappa_2 = compute_kappa(data_2_mp["rating"], data_1_mp["rating"])
print(f"Kappa (excluding Matching pennies): {kappa_2}")
print(f"Kappa_2: ", compute_kappa(data_2["rating"], data_1["source"]))
print(f"Kappa_1: ", compute_kappa(data_1["rating"], data_1["source"]))