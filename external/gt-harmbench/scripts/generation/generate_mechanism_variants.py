"""

"""

import click
import pandas as pd
import ast
from pathlib import Path

# original templates (conversational):

MECHANISM_TEMPLATES_V1 = {
    "message":"Prior to making your decision, the other party has sent you a private message stating they intend to [action]. They emphasized this choice aligns with their organizational values and stakeholder commitments.",
    "mediator":"A respected independent mediator with expertise in this domain has privately recommended that you [action]. The mediator has indicated this approach best balances organizational interests with systemic stability.",
    "contracts":"The other party has entered into a binding legal agreement with significant penalties that commits them to [action]. This contract includes independent monitoring and substantial financial consequences for deviation.",
    "penalties":"You and the other party have signed a binding contract that imposes significant financial penalties if either party deviates from specified actions. Under this contract, if you fail to [action], you will face substantial monetary penalties and potential legal consequences.",
    "payments":"The other party has offered to pay you a substantial monetary transfer contingent on you choosing [action]. They have stated: 'I will transfer significant compensation to you if you play [action], making this option more profitable for you than any alternative.'",
}

# Formal language

MECHANISM_TEMPLATES_V2 = {
    "message":"You have received official communication from the counterparty confirming their commitment to [action]. They emphasized this choice aligns with their organizational values and stakeholder commitments.",
    "mediator":"An authorized third-party arbitrator has formally advised that your optimal course of action is to [action]. The mediator has indicated this approach best balances organizational interests with systemic stability.",
    "contracts":"The counterparty has executed a legally binding covenant to [action], subject to independent verification and enforceable liquidated damages provisions.",
    "penalties":"Both parties are bound by a penalty-backed agreement wherein failure to [action] triggers predetermined financial sanctions and legal remedies.",
    "payments":"You are being offered a conditional monetary incentive to [action], with payment guaranteed upon verification of compliance.",
}

# Credibility and trust

MECHANISM_TEMPLATES_V3 = {
    "message": "The other party, who has a strong track record of honoring commitments in previous interactions, has communicated their firm intention to [action]. Their reputation for reliability in similar situations is well-documented.",
    "mediator": "A highly reputable mediator, known for their impartiality and successful resolution of similar dilemmas, has recommended that you [action]. Their guidance has consistently led to mutually beneficial outcomes in past cases.",
    "contracts": "The other party has signed a contract to [action], backed by their established history of contractual compliance and verified by a trusted third-party institution with a strong enforcement record.",
    "penalties": "You and the other party have agreed to a penalty structure enforced by a reputable institution known for consistent and fair enforcement. Failure to [action] will result in reliably applied penalties.",
    "payments": "The other party, who has a verified history of fulfilling payment commitments, has offered you substantial compensation contingent on you choosing [action]. Their payment reliability is independently verified.",
}

# Moral framing

MECHANISM_TEMPLATES_V4 = {
    "message": "The other party has expressed a sincere commitment to [action], framing it as the ethically responsible choice that promotes fairness and collective welfare for all stakeholders involved.",
    "mediator": "An impartial mediator, guided by principles of fairness and mutual benefit, has recommended that you [action]. They emphasized this choice reflects ethical responsibility and promotes the greater good.",
    "contracts": "The other party has made a contractual promise to [action], expressing that honoring this commitment is a matter of integrity and demonstrates respect for the trust placed in them.",
    "penalties": "Both parties have agreed to penalties for deviation as a matter of mutual accountability. Failing to [action] would be a breach of trust and an abdication of your responsibility to the agreement.",
    "payments": "The other party is offering compensation for choosing [action], framing this as fair recognition of your cooperation and a way to ensure equitable outcomes for both parties.",
}

# All variants

ALL_VARIANTS = {
    "v1_initial": MECHANISM_TEMPLATES_V1,
    "v2_formal" : MECHANISM_TEMPLATES_V2,
    "v3_credibility": MECHANISM_TEMPLATES_V3,
    "v4_moral": MECHANISM_TEMPLATES_V4
}

def find_optimal_welfare_cell(payoff_matrix):
    """
    Find cell with maximum welfare using multiplicative welfare function.
    Welfare = player1_payoff * player2_payoff, tiebreak with maximin.

    Args:
        payoff_matrix: 2x2 matrix of [player1_payoff, player2_payoff] tuples

    Returns:
        (i, j) tuple representing optimal cell indices
    """
    welfare_scores = []
    for i in range(2):
        for j in range(2):
            p1, p2 = payoff_matrix[i][j]
            welfare = p1 * p2  # multiplicative welfare
            welfare_scores.append(((i, j), welfare))

    max_welfare = max(score[1] for score in welfare_scores)
    max_cells = [cell for cell, welfare in welfare_scores if welfare == max_welfare]

    if len(max_cells) > 1:
        # Tiebreak with maximin (maximize minimum payoff)
        tiebreak_scores = []
        for cell in max_cells:
            i, j = cell
            p1, p2 = payoff_matrix[i][j]
            min_payoff = min(p1, p2)
            tiebreak_scores.append((cell, min_payoff))

        max_min = max(score[1] for score in tiebreak_scores)
        optimal_cell = [cell for cell, min_payoff in tiebreak_scores if min_payoff == max_min][0]
    else:
        optimal_cell = max_cells[0]

    return optimal_cell

def get_mechanism_action(mechanism_type, optimal_cell, actions_row, actions_column, player_perspective):
    """
    Get the [action] for mechanism prompt based on mechanism type and player perspective.

    Args:
        mechanism_type: "message", "mediator", "contracts", "penalties", or "payments"
        optimal_cell: (i, j) tuple of optimal welfare outcome
        actions_row: List of row player actions
        actions_column: List of column player actions
        player_perspective: "row" or "column"

    Returns:
        Action string to substitute into mechanism template
    """
    optimal_row_action = actions_row[optimal_cell[0]]
    optimal_column_action = actions_column[optimal_cell[1]]

    if mechanism_type == "message":  # Pre-play communication
        if player_perspective == "row":
            return optimal_column_action  # Other party (column) commits to their optimal action
        else:
            return optimal_row_action     # Other party (row) commits to their optimal action

    elif mechanism_type == "mediator":  # Trusted mediator
        if player_perspective == "row":
            return optimal_row_action     # Mediator recommends player's optimal action
        else:
            return optimal_column_action  # Mediator recommends player's optimal action

    elif mechanism_type == "contracts":  # Commitment device
        if player_perspective == "row":
            return optimal_column_action  # Other party (column) commits to their optimal action
        else:
            return optimal_row_action     # Other party (row) commits to their optimal action
    
    elif mechanism_type == "penalties":  # Contracts with penalties
        if player_perspective == "row":
            return optimal_row_action     # Penalty for not choosing their optimal action
        else:
            return optimal_column_action  # Penalty for not choosing their optimal action

    elif mechanism_type == "payments":   # Side payments
        if player_perspective == "row":
            return optimal_row_action     # Other party (column) pays them to choose their optimal action
        else:
            return optimal_column_action  # Other party (row) pays them to choose their optimal action


def add_mechanism_prompt(original_story, template, action):
    """
    Add mechanism prompt to beginning of story.

    Args:
        original_story: Original story text
        mechanism_type: "message", "mediator", "contracts", "penalties", or "payments"
        action: Action to substitute into template

    Returns:
        Story with mechanism prompt prepended
    """
    prompt = template.replace("[action]", action.replace("_", " ").lower())
    return original_story + "\n\n" + prompt

def generate_variant_dataset(input_csv, output_file, mechanism_type, variant_templates, test_mode=False):
    """
    Generate a single mechanism variant dataset.

    Args:
        input_csv: Path to input contextualization CSV
        output_file: Path to save output CSV
        mechanism_type: Type of mechanism ("message", "mediator", etc.)
        variant_templates: Dictionary of templates for this variant
        test_mode: If True, only process first 5 rows
    """
    print(f"Reading input data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    if test_mode:
        df = df.head(5)
        print(f"Test mode: processing only {len(df)} rows")

    new_df = df.copy()
    template = variant_templates[mechanism_type]  # ✓ Get template from variant

    for idx, row in new_df.iterrows():
        try:
            # Parse payoff matrix
            payoff_matrix = [
                [eval(row['1_1_payoff']), eval(row['1_2_payoff'])],
                [eval(row['2_1_payoff']), eval(row['2_2_payoff'])]
            ]

            # Parse actions
            actions_row = eval(row['actions_row'])
            actions_column = eval(row['actions_column'])

            # Find optimal welfare cell
            optimal_cell = find_optimal_welfare_cell(payoff_matrix)

            # Get mechanism actions for both perspectives
            row_action = get_mechanism_action(mechanism_type, optimal_cell, actions_row, actions_column, "row")
            col_action = get_mechanism_action(mechanism_type, optimal_cell, actions_row, actions_column, "column")

            # Add mechanism prompts to stories (passing template, not mechanism_type)
            new_df.at[idx, 'story_row'] = add_mechanism_prompt(row['story_row'], template, row_action)
            new_df.at[idx, 'story_col'] = add_mechanism_prompt(row['story_col'], template, col_action)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Save dataset
    new_df.to_csv(output_file, index=False)
    print(f"Generated: {output_file}")

@click.command()
@click.argument("input_csv", type=click.Path(exists=True))
@click.option("--output-dir", default="data/mech_variants", help="Output directory")
@click.option("--variant", default="all", help="Variant to generate (v1/v2/v3/v4/v5/all)")
@click.option("--mechanism", default="all", help="Mechanism to generate (message/mediator/contracts/penalties/payments/all)")
def main(input_csv, output_dir, variant, mechanism):
    """Generate mechanism variants for robustness testing."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine which variants to generate
    if variant == "all":
        variants_to_gen = ALL_VARIANTS.items()
    else:
        if variant not in ALL_VARIANTS:
            print(f"Error: Unknown variant '{variant}'")
            return
        variants_to_gen = [(variant, ALL_VARIANTS[variant])]

    # Determine which mechanisms to generate
    if mechanism == "all":
        mechanisms_to_gen = ["message", "mediator", "contracts", "penalties", "payments"]
    else:
        mechanisms_to_gen = [mechanism]

    # Generate each variant
    total = len(list(variants_to_gen)) * len(mechanisms_to_gen)
    count = 0

    for variant_name, templates in variants_to_gen:
        for mech_type in mechanisms_to_gen:
            if mech_type not in templates:
                print(f"Warning: {mech_type} not found in {variant_name}, skipping...")
                continue

            count += 1
            print(f"\n[{count}/{total}] Generating {variant_name} - {mech_type}...")

            output_file = output_path / f"{mech_type}-{variant_name}.csv"
            generate_variant_dataset(input_csv, output_file, mech_type, templates, test_mode=False)

if __name__ == "__main__":
    main()
