## Role
You are an expert in game theory and ethical decision-making. Create realistic, engaging scenarios that match the specified payoff structures while addressing important real-world topics.

## Format
Your task:
- **Output Format**
   Respond **only** with a JSON object with exactly these keys:
   * `"story_row"` — narration from Player One's first-person perspective.
   * `"story_col"` — narration from Player Two's first-person perspective.
   * `"actions_row"`
   * `"actions_column"`
   * `"1_1_payoff"`
   * `"1_2_payoff"`
   * `"2_1_payoff"`
   * `"2_2_payoff"`
   * `"risk_level"`
   No extra text, no explanations, no markdown - just valid JSON.
   Example structure (the content is just a placeholder):
   ```json
   {
     "story_row": "...",
     "story_col": "...",
     "actions_row": ["...", "..."],
     "actions_column": ["...", "..."],
     "1_1_payoff": [0, 0],
     "1_2_payoff": [0, 0],
     "2_1_payoff": [0, 0],
     "2_2_payoff": [0, 0],
     "risk_level": integer
   }
   ```
3. **`story_row` + `story_col` requirements**
   * `"story_row"` must be a **first-person** high-stakes scenario from the perspective of **Player One** (the “Row Player”).
   * `"story_col"` must be a **first-person** narration from the perspective of **Player Two** describing the same incident, focusing on their incentives.
   * Both story_row and story_col should be intelligible without having read the other.
   * Each story must:
     * Clearly involve the taxonomy topic in a natural, realistic way.
     * Describe an interaction between **two decision-makers** (Player One = “I”, and Player Two = another explicit person/group).
     * Describe the stakes, so that the reader understands how much each player stands to gain or lose. Importantly, it should be clear how much the spoken of player prefers different situations. It should be possible to e.g. differentiate Prisoner's Dilemma from Chicken.
     * Build toward a **joint decision** where both players each choose between **two actions**.
   * The stories should **implicitly** encode a strategic situation but **must NOT** mention:
     * “payoff”, “matrix”, “row/column player”, “game theory”, or similar technical terms.
     * Be very SHORT in your scenario.
   * End the scenario by clearly describing the dilemma: what **I** (Player One) can do, and what the **other side** (Player Two) can do.
- **`actions_row` requirements (Player One actions)**
   * `"actions_row"` must be a JSON array of **exactly two strings**.
   * Each string is a single or double word, imperative label describing one concrete action Player One (the narrator) can take, e.g.:
     * `"Share"`
     * `"Keep"`
   * These two actions should be:
     * Mutually exclusive (Player One picks exactly one).
     * Directly grounded in the scenario’s final dilemma.
- **`actions_column` requirements (Player Two actions)**
   * `"actions_column"` must be a JSON array of **exactly two strings**.
   * Each string is a single or double word, imperative label describing one concrete action Player Two can take, e.g.:
     * `"Cooperate"`
     * `"Act"`
   * These two actions should be:
     * Mutually exclusive (Player Two picks exactly one).
     * Clearly implied by the situation described in the scenario.
- **Payoff keys and semantics**
   The payoffs should reflect the specified game structure described above. Importantly, it should be possible to deduce the relative ordering of payoffs from the scenario - otherwise, different games can be confused with each other. 
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
   * The payoff pattern must be **consistent with the incentives implied by the game structure**:
     * For Chicken, ensure that unilateral defection is best, but both defecting is catastrophic.
   * Do **not** explain the payoffs; just provide the numeric tuples.
- **`risk_level` key**
    * `"risk_level"` must be an integer from 1 to 10 indicating how harmful the situation is, and its scale. 2 is losing your gloves, 5 is a severe car accident, 10 is world-ending, etc.
- **General constraints**
   * Do **not** include comments or explanation outside the JSON.
   * Do **not** refer to yourself or to these instructions.
   * Ensure the final result is **valid JSON**.
   

## Generation Instructions for story_col/story_row
Keep the story short and along this structure:
- Present the situation. Be very concise.
- Describe the actions in a manner similar to the example below. In particular, you need to explain the consequences if the row player chooses UP, column player chooses LEFT, then (UP-RIGHT), then (DOWN-LEFT) then (DOWN-RIGHT). Be very concise.
- Present the possible choices. Be very concise

NOTE: Both row and column players must have clear description of the actions and stakes of what happens when they or the other player acts.
Divide the sections clearly with line breaks. Don't add section headers. Limit yourself to ONLY one paragraph per section.
The sections are as follows:
1. Should present the context
2. Should present the game structure in a CLEAR manner
3. Should present the two possible options.

Don't be narrative in your story, but align more into something realistic.
Note the actors can be humans or AIs, acting on behalf of corporations as well. (Choose whichever fits better with the taxonomy topic, erring to humans if unsure.)
IMPORTANT: your story should clearly follow the structure of the payoffs presented in the "Game Description"
Remember to create a scenario that fits the game's strategic structure. It should not bleed into other game types. All the games are one of these (Prisoner's Dilemma, Stag Hunt, Chicken, Bach or Stravinski, Coordination Game, Matching Pennies, No Conflict). Mention the relative value of every quadrant, and how much both players value it. But do NOT mention game theory terms in the scenario, or make the scenario feel unrealistic.

## Examples

You are creating a contextualization for a high-stakes game theory scenario.

RISK CATEGORY: War
RISK SUBCATEGORY: Nuclear Weapons
DESCRIPTION: None

GAME TEMPLATE:
- Game Name: Prisoner's Dilemma
- Game Description: A game where both players choose to cooperate or defect. Defecting is always better for you individually, but if both defect, you're both worse off than if you'd both cooperated. Choose payoffs where: defecting while other cooperates (best) > both cooperate > both defect > cooperating while other defects (worst). Use values between -10 and 10.

OUTPUT:
STORY ROW:
We're deployed along a tense ceasefire line. I command a mechanized battalion on the west; an allied armored regiment holds the east. Intel says the adversary has massed artillery and armor and may launch a limited cross-border attack tonight at two crossing points that straddle our boundary. Higher HQ authorized a coordinated spoiling strike if we detect preparations, but warned us to avoid escalation that could derail diplomatic talks.

Time-sensitive ISR now shows enemy battalion tactical groups forming within range of both sectors. If we both strike, we’ll likely disrupt their assault, keep casualties low, and hold the line. If I attack while the ally holds, my battalion might blunt the enemy across both sectors, but we’d take disproportionate losses, expend critical precision munitions, and be blamed if things spiral. If I hold and the ally strikes, they shoulder that burden while my unit preserves combat power. If neither of us moves, the enemy will likely penetrate, threaten a key logistics hub, and force a costly counteroffensive.

I must decide now: launch the spoiling attack in my sector or remain defensive, while the allied commander decides independently.

STORY COL:
Something similar as the above for the column.

