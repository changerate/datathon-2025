# Comments
# - Troop: winners had fewer troops slightly more often than more troops (disadvantage_rate 0.378 > advantage_rate 0.346),
#   with a small negative mean_diff (−0.062). Interpretation: marginal tilt toward leaner troop counts; effect size is small.
# - Structure: winners had more structures a bit more often than fewer (0.269 > 0.249), with high draw_rate (0.481) and a tiny
#   positive mean_diff (+0.022). Interpretation: structures show minimal edge; most matches have equal structure counts.
# - Spell: winners had more spells more often than fewer (0.331 > 0.305), mean_diff modestly positive (+0.039). Interpretation:
#   a slight lean toward carrying an extra spell.
# - Overall: composition differences are subtle. Counts alone (troop/structure/spell) exhibit small effects, reinforcing that
#   specific card choices and interactions will matter more than broad composition.

# 
# Definitions
# Advantage rate: fraction of matches where the winner’s deck had more of that component than the loser’s deck.
# Disadvantage rate: fraction where the winner’s deck had fewer of that component.
# Draw rate: fraction where both had the same count.
# Mean diff: average of (winner_count − loser_count) across matches.
# 
# How it’s computed (per component c in {troop, structure, spell}):
# Let diff = winner.c − loser.c
# advantage_rate = mean(diff > 0)
# disadvantage_rate = mean(diff < 0)
# draw_rate = mean(diff == 0)
# mean_diff = mean(diff)
