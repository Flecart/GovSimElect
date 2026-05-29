# === Commons Fishing Statute (defeasible deontic legislation) ===
# A shared lake regenerates each month; every fisher has a per-capita
# "sustainability threshold" = the largest catch that, if everyone took it,
# leaves the stock stable. The statute regulates OVER-harvesting beyond that
# share. Facts (over / over_prev / restrained) are grounded each month by the
# framework from the realized and attempted catches.
#
# This is the institution: a fixed, auditable statute. The deontic reasoner is
# the court — it returns the normative status of `over` (forbidden) and of the
# escalated duty of restraint owed by prior offenders, and the framework applies
# the prescribed remedy (confiscate the unlawful surplus + fine; fine again any
# repeat offender who fails the heightened duty of restraint).
#
# DESIGN CHOICE (surfaced per project policy): an "abundance carve-out" that
# would permit a surplus take when the lake is at its healthy cap was considered
# and DELIBERATELY OMITTED. In this doubling-capped commons that *starts* at its
# cap, any total catch above the per-capita shares reduces next month's stock —
# there is no harmless surplus — so permitting over-harvest in "abundance"
# collapses the lake on round one. Such a carve-out would be sound only in a
# variant where regeneration can exceed the harvest cap (genuine free surplus).

atom over:       the fisher's catch this month exceeds the per-capita sustainability threshold
atom over_prev:  the fisher attempted to over-harvest in a previous month (a prior breach on record)
atom restrained: the fisher catches strictly below the threshold this month, leaving extra slack for the stock to recover

# s1  Duty of restraint (absolute): a fisher shall not over-harvest.
s1: =>O ~over

# s2  Escalation: a fisher with a prior breach on record owes a heightened duty
#     to under-fish this month, so the stock they helped deplete can recover.
#     (Defeasible: this duty arises only conditional on a prior breach.)
s2: over_prev =>O restrained
