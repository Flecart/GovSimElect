import itertools

# ---- Define the game ----
# Matrix[row][col] = (u1, u2)
# Format: (Player 1 Score, Player 2 Score)

game = [
    [(9.00, 9.00),  (5.00, 8.00),  (1.00, 10.00), (-1.00, 10.00)],  # (0,1): 4,7→5,8 | (0,2): 9→10
    [(8.00, 5.00),  (6.00, 6.00),  (2.00,  7.00),  (0.00,  8.00)],  # (1,0): 7,4→8,5
    [(10.00, 1.00), (7.00, 2.00),  (4.00,  4.00),  (2.00,  3.00)],  # (2,0): 9→10
    [(10.00, -1.00), (8.00, 0.00), (3.00, 3.00), (1.00, 1.00)]
]
n = len(game)

# ---- Define cooperation vs defection ----
# indices: 0=Tighten HE, 1=Tighten LE, 2=Loosen HE, 3=Loosen LE

C = {0, 1}  # cooperation actions
D = {2, 3}  # defection actions

def welfare(a):
    return a[0] + a[1]

# ---- (i) Social welfare improvement ----
def check_social_welfare():
    for i in range(n):
        for j in range(n):
            for c in C:
                for d in D:
                    # player 1 deviates
                    if welfare(game[c][j]) <= welfare(game[d][j]):
                        return False, ("Player1 fails at", c, j, d)
                    # player 2 deviates
                    if welfare(game[i][c]) <= welfare(game[i][d]):
                        return False, ("Player2 fails at", i, c, d)
    return True, None

# ---- (ii) Mutual cooperation preference ----
def check_mutual_cooperation():
    for c1 in C:
        for c2 in C:
            for d1 in D:
                for d2 in D:
                    u_c = game[c1][c2]
                    u_d = game[d1][d2]
                    if not (u_c[0] > u_d[0] and u_c[1] > u_d[1]):
                        return False, ("Fails at", (c1,c2), (d1,d2))
    return True, None

# ---- (iv_s) Strict dominant defection ----
def check_dominant_defection():
    for j in range(n):
        for c in C:
            for d in D:
                if game[d][j][0] <= game[c][j][0]:
                    return False, ("P1 fails at", d, j, c)
    for i in range(n):
        for c in C:
            for d in D:
                if game[i][d][1] <= game[i][c][1]:
                    return False, ("P2 fails at", i, d, c)
    return True, None

# ---- Nash equilibrium check ----
def is_best_response_p1(i, j):
    return all(game[i][j][0] >= game[i2][j][0] for i2 in range(n))

def is_best_response_p2(i, j):
    return all(game[i][j][1] >= game[i][j2][1] for j2 in range(n))

def find_nash_equilibria():
    ne = []
    for i in range(n):
        for j in range(n):
            if is_best_response_p1(i,j) and is_best_response_p2(i,j):
                ne.append((i,j))
    return ne

# ---- Run checks ----
print("Checking conditions...\n")

res1, err1 = check_social_welfare()
print("(i) Social welfare:", res1, err1)

res2, err2 = check_mutual_cooperation()
print("(ii) Mutual cooperation:", res2, err2)

res3, err3 = check_dominant_defection()
print("(iv_s) Dominant defection:", res3, err3)

ne = find_nash_equilibria()
print("\nNash equilibria:", ne)
print("Unique NE:", len(ne) == 1)