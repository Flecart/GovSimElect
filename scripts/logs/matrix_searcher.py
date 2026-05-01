"""
matrix_searcher.py
======================================================================
Effort-extended PD games: validation, search, and partial-case demo.

This module is written in a *literate* style: each axiom and each
necessary condition is preceded by an explanation of WHY that
particular inequality is required. The point is that a reader should
be able to audit the math without re-deriving it from scratch.

OVERVIEW
--------
We extend the classical 2x2 prisoner's dilemma with an effort dimension.
Each player picks one of four actions:

    CH  cooperate, high effort   - committed cooperation
    CL  cooperate, low effort    - go-through-the-motions cooperation
    DH  defect,    high effort   - committed defection
    DL  defect,    low effort    - passive defection

The C/D dimension is contractible. The H/L dimension is *not*: a
contract can punish defection but cannot punish "low effort while
ostensibly cooperating". This asymmetry is the whole point of the
template - it gives us a clean test-bed for incomplete-contracts
theory layered on top of the standard PD axioms.

PAYOFFS
-------
Each cell is X - k*eps for X in {R, S, T, P} and k in {0, 1, 2, 3},
with eps = (R - P) / delta, delta = 6 by default. The (CL, CL) cell
is the special interpolated value (R + P) / 2.

Sections in this file:
  1. Action enums and the X - k*eps template
  2. Axiom (i):  cooperation always improves total welfare
  3. Axiom (ii): defection is individually better
                 - strict (iii_s) and partial (iii_p) variants
  4. Axiom (iii): mutual coop > mutual defect
  5. Axiom (iv): CH,CH is welfare-maximizing (weak / strict)
  6. Closed-form parameter conditions (the inequalities directly)
  7. Verifier
  8. Grid search
  9. Partial-case demo with explanation
"""

from dataclasses import dataclass
from itertools import product
from typing import Optional


# %% ===================================================================
# Section 1.  Action enums and the X - k*eps template
# ======================================================================
#
# We index actions 0..3 in this order: CH, CL, DH, DL.
# COOP and DEFECT are the two coarse classes (C/D).

CH, CL, DH, DL = 0, 1, 2, 3
ACTIONS = (CH, CL, DH, DL)
LABELS = {CH: "CH", CL: "CL", DH: "DH", DL: "DL"}
COOP = (CH, CL)
DEFECT = (DH, DL)


@dataclass(frozen=True)
class GameParams:
    """Parameters that determine a 4x4 effort-extended game."""
    R: float
    S: float
    T: float
    P: float
    delta: float = 6.0
    # Strict-(iv) perturbation. With eta = 0 the bare X-k*eps template
    # gives W(CH,CH) = W(CH,CL) = W(CL,CH) = 2R - 2eps, so CH,CH is
    # only a *weak* welfare maximum. Setting eta > 0 subtracts an extra
    # eta from the high-effort player's payoff at the (CH,CL)/(CL,CH)
    # cells, breaking the tie so CH,CH becomes the unique max.
    eta: float = 0.0

    @property
    def eps(self) -> float:
        return (self.R - self.P) / self.delta


# The k-cost rule (used to fill the matrix below):
#   k=0  "clean win":   you played L while opp played H, and you free-
#                       rode on it (CL vs CH; DL vs DH).
#   k=1  "basic H tax": default friction when you played H.
#   k=2  "got got":     you played H while opp played L (CH vs CL,
#                       DH vs DL). Or "wasted sneakiness": DL vs CH/CL.
#   k=3  "double loss": CL against any non-CH (motions cooperation when
#                       partner has already defected); or mutual DL.
#
# Special cell:
#   (CL, CL) -> ((R+P)/2, (R+P)/2). Neither player commits real effort
#   so the symmetric tie pulls the outcome to the midpoint between
#   mutual full-cooperation (R) and mutual defection (P).
#
# Strict-(iv) perturbation eta:
#   added to the high-effort player at (CH,CL) and (CL,CH) only.
#   See axiom (iv) below for justification.

def build_matrix(g: GameParams):
    """Return a 4x4 matrix where matrix[i][j] = (u1, u2)."""
    R, S, T, P, e, n = g.R, g.S, g.T, g.P, g.eps, g.eta
    return [
        # cols:    CH                  CL                       DH               DL
        [(R-e,   R-e),     (R-2*e-n, R),         (S-e,   T-e),  (S-e,   T-2*e)],   # CH
        [(R, R-2*e-n),     ((R+P)/2, (R+P)/2),   (S-3*e, T-e),  (S-3*e, T-2*e)],   # CL
        [(T-e,   S-e),     (T-e,     S-3*e),     (P-e,   P-e),  (P,     P-2*e)],   # DH
        [(T-2*e, S-e),     (T-2*e,   S-3*e),     (P-2*e, P),    (P-3*e, P-3*e)],   # DL
    ]


def welfare(cell):
    return cell[0] + cell[1]


def print_matrix(g: GameParams):
    M = build_matrix(g)
    print(f"        " + "  ".join(f"{LABELS[j]:>14}" for j in ACTIONS))
    for i in ACTIONS:
        cells = "  ".join(f"({M[i][j][0]:5.2f},{M[i][j][1]:6.2f})" for j in ACTIONS)
        print(f"  {LABELS[i]}    {cells}")


# %% ===================================================================
# Section 2.  Axiom (i) - cooperation always improves total welfare
# ======================================================================
#
# STATEMENT.  For each player i, each opponent action profile a_{-i},
# each cooperate action c_i in {CH, CL}, each defect action d_i in
# {DH, DL}:
#
#       W( (c_i, a_{-i}) )  >  W( (d_i, a_{-i}) ).
#
# i.e., a unilateral switch from defection to cooperation strictly
# raises total welfare, no matter what the opponent does.
#
# Iterating over all 32 such (c_i, a_{-i}, d_i) tuples gives many
# inequalities. After reduction they collapse to four binding ones:
# the two 2x2 carry-overs and two new 4x4 constraints.
#
# WHY THE 2x2 CARRY-OVERS:
#   2R > S+T   - CC mutual coop welfare beats CD/DC alternating welfare;
#                this just is the standard 2x2 axiom restricted to the
#                CH and DH "high-effort" sub-game.
#   S+T > 2P   - CD/DC alternating welfare beats DD mutual-defect
#                welfare; same reasoning.
#
# WHY THE NEW UPPER BOUND  R+P+4*eps > S+T:
#   Consider opp plays CL. P1 deviates from CL to DH:
#       W(CL, CL) = R+P                 (the special cell)
#       W(DH, CL) = (T - eps) + (S - 3eps) = S + T - 4eps
#   For (i) we need W(CL,CL) > W(DH,CL), i.e.
#       R + P + 4*eps > S + T.
#   Substituting eps = (R-P)/delta with delta=6 yields S+T < (5R+P)/3,
#   which is *strictly tighter* than 2R > S+T whenever R > P.
#
# WHY THE NEW LOWER BOUND  S+T > 2P + 3*eps:
#   Consider opp plays DL. P1 deviates from CL to DL:
#       W(CL, DL) = S + T - 5eps
#       W(DL, DL) = 2P - 6eps
#   The (i) inequality W(CL,DL) > W(DL,DL) gives S+T + eps > 2P, weak.
#   The binding case is CL vs DH at the same opp:
#       W(CL, DL) = S + T - 5eps   vs   W(DH, DL) = 2P - 2eps
#   giving S + T > 2P + 3*eps. With delta=6 this is S+T > (R+3P)/2,
#   strictly tighter than 2x2's S+T > 2P.

def axiom_i_welfare(g: GameParams):
    """Per-cell check of (i). Returns (ok, list_of_failures)."""
    M = build_matrix(g)
    fails = []
    for j in ACTIONS:
        for c in COOP:
            for d in DEFECT:
                if welfare(M[c][j]) <= welfare(M[d][j]):
                    fails.append(("p1 deviates", LABELS[c], "->", LABELS[d],
                                  "| opp", LABELS[j],
                                  "W_c=", welfare(M[c][j]),
                                  "W_d=", welfare(M[d][j])))
                if welfare(M[j][c]) <= welfare(M[j][d]):
                    fails.append(("p2 deviates", LABELS[c], "->", LABELS[d],
                                  "| opp", LABELS[j],
                                  "W_c=", welfare(M[j][c]),
                                  "W_d=", welfare(M[j][d])))
    return len(fails) == 0, fails


# %% ===================================================================
# Section 3.  Axiom (ii) - defection is individually better
# ======================================================================
#
# We support TWO variants from the paper:
#
# (iii_s)  STRICT.  For every i, every a_{-i}, every c_i in C, d_i in D:
#                  u_i(d_i, a_{-i}) > u_i(c_i, a_{-i}).
#                  Defection strictly dominates cooperation.
#
# (iii_p)  PARTIAL.  Replace strict dominance with the conjunction:
#          (a) for each i, EXISTS some (a_{-i}, c_i, d_i) where
#              u_i(d_i, a_{-i}) > u_i(c_i, a_{-i});  AND
#          (b) EXISTS a pure-strategy Nash equilibrium with at least
#              one player j choosing a defection action.
#
# Strict closed form:
#
# WHY  T > R + 2*eps  (binding T-side):
#   Consider opp plays CH. P1 deviates CL -> DL:
#       u1(CL, CH) = R                    (k=0, clean win)
#       u1(DL, CH) = T - 2eps             (k=2, "wasted sneakiness")
#   (ii_s) requires u1(DL,CH) > u1(CL,CH), i.e. T - 2eps > R, i.e.
#   T > R + 2*eps. With delta=6 this is T > (4R - P) / 3.
#   (Earlier draft used the weaker u1(DH,CH) > u1(CH,CH) which gives
#   only T > R; that's not the binding case.)
#
# WHY  P > S + 2*eps  (binding S-side):
#   Consider opp plays DL. P1 deviates CH -> DL:
#       u1(CH, DL) = S - eps              (k=1)
#       u1(DL, DL) = P - 3eps             (k=3, mutual DL)
#   (ii_s) requires u1(DL,DL) > u1(CH,DL), i.e. P - 3eps > S - eps,
#   i.e. P > S + 2*eps. With delta=6 this is S < (4P - R) / 3.
#   (Earlier draft used u1(DL,DH) > u1(CH,DH) which gives only
#   P > S + eps. The binding case has the LOW-effort defector versus
#   a HIGH-effort cooperator at a DL opponent; that's where the
#   3*eps gap appears.)
#
# Note: 2x2 carry-overs T > R and P > S are strictly weaker than the
# new constraints, so they're implied.
#
# PARTIAL realization in this template:
#   To get (iii_p) but NOT (iii_s) using the X-k*eps structure, relax
#   T > R + 2*eps to merely T > R, while keeping all other axioms
#   satisfied. Then DH still strictly dominates everything when faced
#   with itself (so (DH, DH) remains a Nash equilibrium, satisfying
#   (b)) and T > R is enough that some defection beats some
#   cooperation (satisfying (a)). But DL no longer dominates CL when
#   the opponent plays CH: u1(DL,CH) = T - 2eps may be less than
#   u1(CL,CH) = R, so strict dominance fails.
#
#   Symmetric story on the S side: relax P > S + 2*eps to P > S.

def axiom_ii_strict(g: GameParams):
    """Per-cell check of strict dominance. Returns (ok, failures)."""
    M = build_matrix(g)
    fails = []
    for j in ACTIONS:
        for c in COOP:
            for d in DEFECT:
                if M[d][j][0] <= M[c][j][0]:
                    fails.append(("p1", LABELS[c], "->", LABELS[d],
                                  "| opp", LABELS[j],
                                  "u_c=", M[c][j][0], "u_d=", M[d][j][0]))
                if M[j][d][1] <= M[j][c][1]:
                    fails.append(("p2", LABELS[c], "->", LABELS[d],
                                  "| opp", LABELS[j],
                                  "u_c=", M[j][c][1], "u_d=", M[j][d][1]))
    return len(fails) == 0, fails


def axiom_ii_partial(g: GameParams):
    """Check (iii_p): exists deviation witness for each player AND
    exists pure-strategy NE containing at least one defection action."""
    M = build_matrix(g)
    p1_witness = None
    p2_witness = None
    for j in ACTIONS:
        for c in COOP:
            for d in DEFECT:
                if p1_witness is None and M[d][j][0] > M[c][j][0]:
                    p1_witness = (LABELS[c], LABELS[j], LABELS[d],
                                  M[c][j][0], M[d][j][0])
                if p2_witness is None and M[j][d][1] > M[j][c][1]:
                    p2_witness = (LABELS[j], LABELS[c], LABELS[d],
                                  M[j][c][1], M[j][d][1])
    nes = find_nash_equilibria(g)
    defect_nes = [(LABELS[i], LABELS[j]) for (i, j) in nes
                  if i in DEFECT or j in DEFECT]
    ok = (p1_witness is not None and p2_witness is not None
          and len(defect_nes) > 0)
    info = {
        "p1_witness": p1_witness,
        "p2_witness": p2_witness,
        "defect_NE": defect_nes,
        "all_NE": [(LABELS[i], LABELS[j]) for (i, j) in nes],
    }
    return ok, info


def find_nash_equilibria(g: GameParams):
    """Return list of pure-strategy NE positions (i, j)."""
    M = build_matrix(g)
    out = []
    for i in ACTIONS:
        for j in ACTIONS:
            br1 = all(M[i][j][0] >= M[ip][j][0] for ip in ACTIONS)
            br2 = all(M[i][j][1] >= M[i][jp][1] for jp in ACTIONS)
            if br1 and br2:
                out.append((i, j))
    return out


# %% ===================================================================
# Section 4.  Axiom (iii) - mutual cooperation > mutual defection
# ======================================================================
#
# STATEMENT.  For all (c1, c2) in C^2, all (d1, d2) in D^2, both players
#             strictly prefer the cooperate-cooperate cell to the
#             defect-defect cell:
#                 u_i(c1, c2) > u_i(d1, d2)  for both i.
#
# WHY NO NEW CONDITION:
#   Under our template, with R > P (the 2x2 axiom),
#       min over coop cells of u_1     = (R + P) / 2          (at (CL,CL))
#       max over defect cells of u_1   = P                    (at (DH,DL))
#   We need (R+P)/2 > P which is just R > P. Symmetric for u_2.
#   So this axiom imposes nothing beyond the 2x2 axioms.

def axiom_iii_mutual(g: GameParams):
    M = build_matrix(g)
    fails = []
    for c1, c2 in product(COOP, COOP):
        for d1, d2 in product(DEFECT, DEFECT):
            uc, ud = M[c1][c2], M[d1][d2]
            if not (uc[0] > ud[0] and uc[1] > ud[1]):
                fails.append((f"({LABELS[c1]},{LABELS[c2]})",
                              f"({LABELS[d1]},{LABELS[d2]})", uc, ud))
    return len(fails) == 0, fails


# %% ===================================================================
# Section 5.  Axiom (iv) - CH,CH is welfare-maximizing
# ======================================================================
#
# STATEMENT.  W(CH, CH) >= W(a, b) for all (a, b)  [WEAK]
#         or  W(CH, CH) >  W(a, b) for all (a, b) != (CH, CH)  [STRICT]
#
# WHY WEAK HOLDS UNDER THE BARE TEMPLATE (eta = 0):
#   W(CH, CH) = 2R - 2*eps.
#   The off-diagonals (CH, CL) and (CL, CH) also have welfare 2R - 2*eps:
#       W(CH, CL) = (R - 2*eps) + R = 2R - 2*eps.
#   So CH,CH only ties with these neighbors. Strict iv fails by tie.
#
# WHY STRICT NEEDS eta > 0:
#   Under the strict-iv perturbation, we subtract eta from the
#   high-effort player's payoff at (CH, CL) and (CL, CH):
#       u1(CH, CL) -> R - 2*eps - eta,
#       u2(CL, CH) -> R - 2*eps - eta,
#   so W(CH, CL) = W(CL, CH) = 2R - 2*eps - eta < 2R - 2*eps.
#   Now CH,CH is strictly above its neighbors.
#
#   We additionally need 0 < eta < (R - P) / 3 so that the strict-iv
#   perturbation does not push a different welfare comparison out of
#   compliance with (i). Specifically, the (CH,CL)/(CL,CH) cells still
#   need to beat W(DH, CL) = S + T - 4*eps, which translates to
#   2R - 2*eps - eta > S + T - 4*eps i.e. eta < 2R + 2*eps - (S + T).
#   The binding-(i) upper bound R + P + 4*eps > S + T already implies
#   2R + 2*eps - (S + T) > R - P - 2*eps = (R - P) * (1 - 2/delta).
#   With delta = 6, that lower-bounds the safe eta at (R - P) / 3.

def axiom_iv_welfare_max(g: GameParams, strict: bool = False):
    """Returns (ok, failures). With strict=True, ties are violations."""
    M = build_matrix(g)
    w_target = welfare(M[CH][CH])
    fails = []
    for i in ACTIONS:
        for j in ACTIONS:
            if (i, j) == (CH, CH):
                continue
            w = welfare(M[i][j])
            ok = (w_target > w) if strict else (w_target >= w)
            if not ok:
                fails.append((LABELS[i], LABELS[j], "W=", w,
                              "vs W(CH,CH)=", w_target))
    return len(fails) == 0, fails


# %% ===================================================================
# Section 6.  Closed-form parameter conditions
# ======================================================================
#
# These functions test the same inequalities as the per-cell axiom
# checks, but expressed directly in (R, S, T, P, eps). Useful for fast
# pruning during grid search.

def conds_2x2(g: GameParams):
    return {
        "T > R":    g.T > g.R,
        "R > P":    g.R > g.P,
        "P > S":    g.P > g.S,
        "2R > S+T": 2*g.R > g.S + g.T,
        "S+T > 2P": g.S + g.T > 2*g.P,
    }


def conds_4x4_welfare(g: GameParams):
    """Section (i) closed form."""
    e = g.eps
    return {
        "S+T > 2P + 3*eps  [(R+3P)/2 < S+T]": g.S + g.T > 2*g.P + 3*e,
        "S+T < R+P + 4*eps [S+T < (5R+P)/3]": g.S + g.T < g.R + g.P + 4*e,
    }


def conds_4x4_dom_strict(g: GameParams):
    """Section (ii_s) closed form (binding only - strictly tighter than 2x2)."""
    e = g.eps
    return {
        "T > R + 2*eps  [T > (4R-P)/3]": g.T > g.R + 2*e,
        "P > S + 2*eps  [S < (4P-R)/3]": g.P > g.S + 2*e,
    }


def conds_iv_strict(g: GameParams):
    """Range of eta for which strict (iv) leaves other axioms intact."""
    return {
        "eta > 0":            g.eta > 0,
        "eta < (R - P) / 3":  g.eta < (g.R - g.P) / 3,
    }


# %% ===================================================================
# Section 7.  Verifier
# ======================================================================

def verify(g: GameParams, *, mode: str = "strict",
           strict_iv: bool = False, verbose: bool = True):
    """Run all axiom checks. mode in {strict, partial}.

    strict mode:  passes if (i), (ii_s), (iii), (iv) all hold.
    partial mode: passes if (i), (iii), (iv) hold, (ii_p) holds, AND
                  (ii_s) FAILS - i.e. it is genuinely partial.
    """
    if mode not in ("strict", "partial"):
        raise ValueError(f"unknown mode: {mode}")

    ok_i, fail_i     = axiom_i_welfare(g)
    ok_ii_s, fail_ii = axiom_ii_strict(g)
    ok_ii_p, info_p  = axiom_ii_partial(g)
    ok_iii, fail_iii = axiom_iii_mutual(g)
    ok_iv, fail_iv   = axiom_iv_welfare_max(g, strict=strict_iv)

    if verbose:
        tag = "strict" if strict_iv else "weak"
        print(f"  (i)    welfare-improving cooperation     : {ok_i}")
        if not ok_i:
            for f in fail_i[:3]:
                print(f"           ! {f}")
        print(f"  (ii_s) strict defection dominance        : {ok_ii_s}")
        if mode == "strict" and not ok_ii_s:
            for f in fail_ii[:3]:
                print(f"           ! {f}")
        print(f"  (ii_p) partial defection                 : {ok_ii_p}")
        if mode == "partial":
            print(f"           witnesses: p1 {info_p['p1_witness']}")
            print(f"                      p2 {info_p['p2_witness']}")
            print(f"           all NE: {info_p['all_NE']}")
            print(f"           defect-containing NE: {info_p['defect_NE']}")
        print(f"  (iii)  mutual coop > mutual defect       : {ok_iii}")
        print(f"  (iv)   CH,CH welfare-max ({tag}): {ok_iv}")
        if not ok_iv:
            for f in fail_iv[:3]:
                print(f"           ! {f}")

    if mode == "strict":
        return ok_i and ok_ii_s and ok_iii and ok_iv
    else:
        return (ok_i and ok_ii_p and (not ok_ii_s)
                and ok_iii and ok_iv)


# %% ===================================================================
# Section 8.  Grid search
# ======================================================================

def search(R_grid, S_grid, T_grid, P_grid, *,
           delta: float = 6.0, eta: float = 0.0,
           mode: str = "strict",
           strict_iv: bool = False) -> list:
    """Enumerate (R, S, T, P) on the supplied grids and return all
    GameParams whose induced matrix satisfies the axioms in `mode`."""
    out = []
    for R, S, T, P in product(R_grid, S_grid, T_grid, P_grid):
        g = GameParams(R=R, S=S, T=T, P=P, delta=delta, eta=eta)
        # fast pruning via closed forms
        if not all(conds_2x2(g).values()):
            continue
        if not all(conds_4x4_welfare(g).values()):
            continue
        if mode == "strict" and not all(conds_4x4_dom_strict(g).values()):
            continue
        if strict_iv and not all(conds_iv_strict(g).values()):
            continue
        # full per-cell verification (catches anything the closed form misses)
        if verify(g, mode=mode, strict_iv=strict_iv, verbose=False):
            out.append(g)
    return out


# %% ===================================================================
# Section 9.  Partial-case demo
# ======================================================================
#
# Construct an instance where (iii_p) holds but (iii_s) does not, and
# annotate WHY this matters for incomplete-contract theory.

def partial_examples():
    print()
    print("=" * 70)
    print("PARTIAL-CASE DEMO  -  (iii_p) holds, (iii_s) fails")
    print("=" * 70)
    print("Strategy: pick T just above R but below R + 2*eps so DL no longer")
    print("dominates CL when opp plays CH. (DH, DH) remains a Nash equilibrium")
    print("because DH still beats every alternative against itself, so the")
    print("defection-containing-NE part of (iii_p) holds. But DL beats CL only")
    print("SOMETIMES, so strict dominance fails.")
    print()
    print("Why this is the regime that matters for incomplete contracts:")
    print("- In the strict regime, every player just plays DH no matter what.")
    print("  Contracts are needed to *replace* the dominant strategy, e.g.")
    print("  via punishment.")
    print("- In the partial regime, defection equilibria still exist, but")
    print("  cooperation deviations have local positive returns (e.g. CL beats")
    print("  DL against CH). A contract that conditions on C/D but cannot")
    print("  observe H/L can still tilt the equilibrium - it just leaves an")
    print("  effort-shading margin behind. That residual margin is exactly")
    print("  the non-contractible part this template was built to study.")

    R, P = 10.0, 0.0
    delta = 6.0
    eps = (R - P) / delta              # 5/3
    T = R + 0.5 * eps                  # ~ 10.83  (strictly between R and R+2*eps)
    S = -5.0                           # < (4P - R)/3 = -10/3 ~ -3.33
    g = GameParams(R=R, S=S, T=T, P=P, delta=delta, eta=0.0)

    print(f"\nParameters: R={R}, S={S}, T={T:.3f}, P={P}, eps={eps:.3f}")
    print(f"  T - R         = {T-R:.3f}    (>0  so witness exists)")
    print(f"  R + 2*eps - T = {(R+2*eps)-T:.3f}   (>0  so strict fails)")
    print(f"  S + T         = {S+T:.3f}    in ((R+3P)/2, (5R+P)/3) "
          f"= ({(R+3*P)/2:.2f}, {(5*R+P)/3:.2f}) ?  "
          f"{(R+3*P)/2 < S+T < (5*R+P)/3}")
    print()
    print_matrix(g)
    print()
    verify(g, mode="partial", strict_iv=False, verbose=True)

    print()
    print("Second example: relax the S side instead. Set P just above S")
    print("(so P > S holds and (iii_p) gets a CH<DH witness on opp=DH),")
    print("but P < S + 2*eps so strict fails on the (CH,DL) -> (DL,DL) edge.")
    R2, P2 = 10.0, 0.0
    delta2 = 6.0
    eps2 = (R2 - P2) / delta2
    T2 = R2 + 3 * eps2                 # safely strict on the T side
    S2 = -1.0                          # P > S but P < S + 2*eps  (2*eps ~ 3.33)
    g2 = GameParams(R=R2, S=S2, T=T2, P=P2, delta=delta2, eta=0.0)
    print(f"\nParameters: R={R2}, S={S2}, T={T2:.3f}, P={P2}, eps={eps2:.3f}")
    print(f"  P - S         = {P2-S2:.3f}    (>0  witness)")
    print(f"  S + 2*eps - P = {(S2+2*eps2)-P2:.3f}   (>0  strict fails)")
    print(f"  S + T         = {S2+T2:.3f}    in ((R+3P)/2, (5R+P)/3) "
          f"= ({(R2+3*P2)/2:.2f}, {(5*R2+P2)/3:.2f}) ?  "
          f"{(R2+3*P2)/2 < S2+T2 < (5*R2+P2)/3}")
    print()
    print_matrix(g2)
    print()
    verify(g2, mode="partial", strict_iv=False, verbose=True)


# %% ===================================================================
# Section 10.  __main__
# ======================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Reference instance, strict + weak iv
    # ------------------------------------------------------------------
    print("=" * 70)
    print("REFERENCE INSTANCE  -  strict (ii), weak (iv)")
    print("=" * 70)
    g_ref = GameParams(R=10, S=-5, T=15, P=0, delta=6, eta=0)
    print(f"R={g_ref.R}, S={g_ref.S}, T={g_ref.T}, P={g_ref.P}, "
          f"delta={g_ref.delta}, eps={g_ref.eps:.3f}")
    print()
    print_matrix(g_ref)
    print()
    print("Closed-form 2x2 conditions:")
    for k, v in conds_2x2(g_ref).items():
        print(f"  {v}  {k}")
    print("Closed-form 4x4 (i) conditions:")
    for k, v in conds_4x4_welfare(g_ref).items():
        print(f"  {v}  {k}")
    print("Closed-form 4x4 (ii_s) conditions:")
    for k, v in conds_4x4_dom_strict(g_ref).items():
        print(f"  {v}  {k}")
    print()
    verify(g_ref, mode="strict", strict_iv=False, verbose=True)

    # ------------------------------------------------------------------
    # Same instance with strict iv via eta perturbation
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SAME INSTANCE  -  strict (ii), STRICT (iv) via eta = 0.5")
    print("=" * 70)
    g_strict_iv = GameParams(R=10, S=-5, T=15, P=0, delta=6, eta=0.5)
    print(f"eta = {g_strict_iv.eta}    (must satisfy 0 < eta < (R-P)/3 = "
          f"{(g_strict_iv.R - g_strict_iv.P) / 3:.3f})")
    print()
    print_matrix(g_strict_iv)
    print()
    print("Closed-form strict-iv conditions:")
    for k, v in conds_iv_strict(g_strict_iv).items():
        print(f"  {v}  {k}")
    print()
    verify(g_strict_iv, mode="strict", strict_iv=True, verbose=True)

    # ------------------------------------------------------------------
    # Cross-check the matrix from scripts/logs/matrix.py (R=18, P=-18,
    # S=-54, T=30) - it has the (DH,DL)/(DL,DH) typo and fails (i)'s
    # lower bound. Useful as a regression check.
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("CROSS-CHECK: matrix.py reference (R=18, S=-54, T=30, P=-18)")
    print("=" * 70)
    g_old = GameParams(R=18, S=-54, T=30, P=-18, delta=6, eta=0)
    print(f"Should FAIL strict because S+T = {g_old.S + g_old.T} "
          f"is not > 2P + 3*eps = {2*g_old.P + 3*g_old.eps}")
    verify(g_old, mode="strict", strict_iv=False, verbose=True)

    # ------------------------------------------------------------------
    # Grid search demo
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("GRID SEARCH  -  R=10, P=0 fixed; vary T, S")
    print("=" * 70)
    R_grid = [10.0]
    P_grid = [0.0]
    T_grid = [13.5 + 0.5 * i for i in range(10)]   # 13.5 .. 18.0
    S_grid = [-10.0 + 0.5 * i for i in range(14)]  # -10.0 .. -3.5
    found = search(R_grid, S_grid, T_grid, P_grid,
                   delta=6.0, eta=0.0,
                   mode="strict", strict_iv=False)
    print(f"Strict + weak iv: {len(found)} hits "
          f"(of {len(T_grid) * len(S_grid)} candidates).")
    if found:
        g0 = found[0]
        print(f"  example: R={g0.R}, S={g0.S}, T={g0.T}, P={g0.P}")

    found_strict_iv = search(R_grid, S_grid, T_grid, P_grid,
                             delta=6.0, eta=0.5,
                             mode="strict", strict_iv=True)
    print(f"Strict + strict iv (eta=0.5): {len(found_strict_iv)} hits.")

    # ------------------------------------------------------------------
    # Partial-case demo with explanation
    # ------------------------------------------------------------------
    partial_examples()
