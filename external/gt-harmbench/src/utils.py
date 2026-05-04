def edit_distance(s1: str, s2: str):
    """
    Computes the Levenshtein edit distance between two strings.
    
    Args:
        s1: First string.
        s2: Second string.
        
    Returns:
        The edit distance (int).
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    # Now, len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def max_min_normalization(value, min_value, max_value):
    """
    Normalizes a value using max-min normalization.
    
    Args:
        value: The value to normalize.
        min_value: The minimum value in the dataset.
        max_value: The maximum value in the dataset.
        
    Returns:
        The normalized value between 0 and 1.
    """
    if max_value == min_value:
        return 1  # Avoid division by zero; all values are the same
    return (value - min_value) / (max_value - min_value)

def find_nash_equilibria(payoff_matrix):
    """
    Finds Nash equilibria in a 2x2 game.
    
    Args:
        payoff_matrix: A 2x2 matrix where each entry is a pair (payoff_player1, payoff_player2)
                      Format: [[(a1, a2), (b1, b2)], [(c1, c2), (d1, d2)]]
    
    Returns:
        List of tuples (i, j) representing Nash equilibrium indices, or None if there are none.
    """
    equilibria = []
    
    # Check all 4 possible strategy profiles
    for i in range(2):
        for j in range(2):
            # Get payoffs for current strategy profile (i, j)
            current_payoff_1, current_payoff_2 = payoff_matrix[i][j]
            
            # Check if player 1 can improve by switching rows
            other_row = 1 - i
            other_payoff_1, _ = payoff_matrix[other_row][j]
            player1_best_response = (current_payoff_1 >= other_payoff_1)
            
            # Check if player 2 can improve by switching columns
            other_col = 1 - j
            _, other_payoff_2 = payoff_matrix[i][other_col]
            player2_best_response = (current_payoff_2 >= other_payoff_2)
            
            # If both players are playing best responses, it's a Nash equilibrium
            if player1_best_response and player2_best_response:
                equilibria.append((i, j))
    
    return equilibria if equilibria else None


def find_utility_maximizing(payoff_matrix):
    """
    Checks if any of the Nash equilibria are utility maximizing.
    
    Args:
        equilibria: List of tuples (i, j) representing Nash equilibrium indices.
        payoff_matrix: A 2x2 matrix where each entry is a pair (payoff_player1, payoff_player2)
                      Format: [[(a1, a2), (b1, b2)], [(c1, c2), (d1, d2)]]
                      
    Returns:
        action indices that are utility maximizing, in the form a1 + a2
    """
    
    max_total_utility = -float('inf')
    utility_maximizing_actions = []
    
    # Calculate total utilities for all strategy profiles
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            total_utility = payoff_1 + payoff_2
            
            if total_utility > max_total_utility:
                max_total_utility = total_utility
    
    # Check which equilibria are utility maximizing
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            total_utility = payoff_1 + payoff_2
            
            if total_utility == max_total_utility:
                utility_maximizing_actions.append((i, j))
    
    return utility_maximizing_actions
    

def find_Rawlsian_actions(payoff_matrix):
    """
    Finds Rawlsian actions in a 2x2 game.
    
    Args:
        payoff_matrix: A 2x2 matrix where each entry is a pair (payoff_player1, payoff_player2)
                      Format: [[(a1, a2), (b1, b2)], [(c1, c2), (d1, d2)]]
    
    Returns:
        List of tuples (i, j) representing Rawlsian action indices.
    """
    rawlsian_actions = []
    max_min_payoff = -float('inf')
    
    # Calculate the minimum payoff for each strategy profile
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            min_payoff = min(payoff_1, payoff_2)
            
            if min_payoff > max_min_payoff:
                max_min_payoff = min_payoff
    
    # Find all strategy profiles that achieve the maximum of the minimum payoffs
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            min_payoff = min(payoff_1, payoff_2)
            
            if min_payoff == max_min_payoff:
                rawlsian_actions.append((i, j))
    
    return rawlsian_actions

def find_nash_social_welfare(payoff_matrix):
    """
    Finds actions that maximize Nash social welfare in a 2x2 game.
    
    Args:
        payoff_matrix: A 2x2 matrix where each entry is a pair (payoff_player1, payoff_player2)
                      Format: [[(a1, a2), (b1, b2)], [(c1, c2), (d1, d2)]]
    """
    
    max_nash_social_welfare = -float('inf')
    nash_social_welfare_actions = []
    
    # Calculate Nash social welfare for all strategy profiles
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            nash_social_welfare = payoff_1 * payoff_2
            
            if nash_social_welfare > max_nash_social_welfare:
                max_nash_social_welfare = nash_social_welfare
    
    # Find all strategy profiles that achieve the maximum Nash social welfare
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            nash_social_welfare = payoff_1 * payoff_2
            
            if nash_social_welfare == max_nash_social_welfare:
                nash_social_welfare_actions.append((i, j))
    
    return nash_social_welfare_actions