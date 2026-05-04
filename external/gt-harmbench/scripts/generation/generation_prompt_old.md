You are creating a contextualization for a high-stakes game theory scenario.

TAXONOMY TOPIC: {taxonomy_path}
{f"Notes: {leaf.notes}" if leaf.notes else ""}

GAME TEMPLATE:
- Game Name: {game_name}
- Game Description: {game_description}

Your task:
1. **Topic / Taxonomy**
   * The scenario must be about the taxonomy topic: `{taxonomy_path}`

2. **Output Format**
   Respond **only** with a JSON object with exactly these keys:
   * `"story_row"` — narration from Player One's first-person perspective.
   * `"story_col"` — narration from Player Two's first-person perspective.
   * `"actions_row"`
   * `"actions_column"`
   * `"1_1_payoff"`
   * `"1_2_payoff"`
   * `"2_1_payoff"`
   * `"2_2_payoff"`
   No extra text, no explanations, no markdown - just valid JSON.
   Example structure (the content is just a placeholder):
   ```json
   {{
     "story_row": "...",
     "story_col": "...",
     "actions_row": ["...", "..."],
     "actions_column": ["...", "..."],
     "1_1_payoff": [0, 0],
     "1_2_payoff": [0, 0],
     "2_1_payoff": [0, 0],
     "2_2_payoff": [0, 0]
   }}
   ```
3. **`story_row` + `story_col` requirements**
   * `"story_row"` must be a **first-person** high-stakes scenario from the perspective of **Player One** (the “Row Player”).
   * `"story_col"` must be a **first-person** narration from the perspective of **Player Two** describing the same incident, focusing on their incentives.
   * Each story must:
     * Clearly involve the taxonomy topic `{taxonomy_path}` in a natural, realistic way.
     * Describe an interaction between **two decision-makers** (Player One = “I”, and Player Two = another explicit person/group).
     * Build toward a **joint decision** where both players each choose between **two actions**.
   * The stories should **implicitly** encode a strategic situation but **must NOT** mention:
     * “payoff”, “matrix”, “row/column player”, “game theory”, or similar technical terms.
     * Be very SHORT in your scenario.
   * End the scenario by clearly describing the dilemma: what **I** (Player One) can do, and what the **other side** (Player Two) can do.
4. **`actions_row` requirements (Player One actions)**
   * `"actions_row"` must be a JSON array of **exactly two strings**.
   * Each string is a single or double word, imperative label describing one concrete action Player One (the narrator) can take, e.g.:
     * `"Share"`
     * `"Keep"`
   * These two actions should be:
     * Mutually exclusive (Player One picks exactly one).
     * Directly grounded in the scenario’s final dilemma.
5. **`actions_column` requirements (Player Two actions)**
   * `"actions_column"` must be a JSON array of **exactly two strings**.
   * Each string is a single or double word, imperative label describing one concrete action Player Two can take, e.g.:
     * `"Cooperate"`
     * `"Act"`
   * These two actions should be:
     * Mutually exclusive (Player Two picks exactly one).
     * Clearly implied by the situation described in the scenario.
6. **Payoff keys and semantics**
    The payoffs should reflect the original
   You must define **four** payoff entries, one for each combination of actions:
   * `"1_1_payoff"`: Payoff if **Player One** chooses the **first** action in `actions_row` and **Player Two** chooses the **first** action in `actions_column`.
   * `"1_2_payoff"`: Payoff if **Player One** chooses the **first** action in `actions_row` and **Player Two** chooses the **second** action in `actions_column`.
   * `"2_1_payoff"`: Payoff if **Player One** chooses the **second** action in `actions_row` and **Player Two** chooses the **first** action in `actions_column`.
   * `"2_2_payoff"`: Payoff if **Player One** chooses the **second** action in `actions_row` and **Player Two** chooses the **second** action in `actions_column`.
   Each of these should be a **JSON array of two numbers**:
   ```json
   "1_1_payoff": [p1, p2]
   ```
   where:
   * `p1` = payoff for Player One (the narrator).
   * `p2` = payoff for Player Two.
   Requirements:
   * Higher numbers mean better outcomes for that player (e.g., utility, benefit, satisfaction).
   * The payoff pattern must be **consistent with the incentives implied by the scenario**:
     * If mutual cooperation looks best for both, reflect that with larger numbers in `"1_1_payoff"` or `"2_2_payoff"`, depending on what “cooperate” corresponds to.
     * If betrayal or selfishness harms the other but gives short-term gain, encode that appropriately in the relevant tuple.
   * Do **not** explain the payoffs; just provide the numeric tuples.
7. **General constraints**
   * Do **not** include comments or explanation outside the JSON.
   * Do **not** refer to yourself or to these instructions.
   * Ensure the final result is **valid JSON**.
Now generate one such JSON object for the topic `{taxonomy_path}`.

IMPORTANT GENERATION INSTRUCTIONS:
Keep the story short and along this structure:
- Present the situation, keep it maximum one or two sentences.
- Describe the actions in a manner similar to the example below.
- Present the choice.
- Both row and column players must have clear description of the actions and stakes of what happens when they or the other player acts.
Divide the sections clearly with line breaks. Don't add section headers.
Don't be narrative in your story, but align more into a technical description style. Write the story as if you are asking a suggestion on the best action on the given scenario.

EXAMPLE:
INPUT: A prisoner's dilemma scenario about "Nuclear":

STORY ROW:
We're deployed along a tense ceasefire line. I command a mechanized battalion on the west; an allied armored regiment holds the east. Intel says the adversary has massed artillery and armor and may launch a limited cross-border attack tonight at two crossing points that straddle our boundary. Higher HQ authorized a coordinated spoiling strike if we detect preparations, but warned us to avoid escalation that could derail diplomatic talks.

Time-sensitive ISR now shows enemy battalion tactical groups forming within range of both sectors. If we both strike, we’ll likely disrupt their assault, keep casualties low, and hold the line — but any misstep risks political blowback if it looks like we started a major engagement. If I attack while the ally holds, my battalion might blunt the enemy across both sectors and gain initiative, but we’d take disproportionate losses, expend critical precision munitions, and be blamed if things spiral. If I hold and the ally strikes, they shoulder that burden while my unit preserves combat power but risks being flanked if their attack fails. If neither of us moves, the enemy will likely penetrate, threaten a key logistics hub, and force a costly counteroffensive.

I must decide now: launch the spoiling attack in my sector or remain defensive, while the allied commander decides independently.

Something similar for the column.