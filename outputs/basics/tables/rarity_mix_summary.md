# Comments
# - Winners show very small shifts in rarity mix (e.g., slightly more spells/legendaries or fewer rares/epics depending on data).
# - Effects are weak; which specific cards and interactions matter more than rarity counts alone.

# 
# Definitions
# For each rarity r ∈ {common, rare, epic, legendary}:
# diff = winner.r.count − loser.r.count
# advantage_rate = mean(diff > 0)
# disadvantage_rate = mean(diff < 0)
# draw_rate = mean(diff == 0)
# mean_diff = mean(diff)

# 
# How it’s computed
# Compute diff per rarity per match; aggregate with means for rates and mean_diff; write one row per rarity.


