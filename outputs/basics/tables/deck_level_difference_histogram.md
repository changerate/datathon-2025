# Comments
# - Histogram of total_level_diff = winner.totalcard.level − loser.totalcard.level.
# - Big bar near 0 shows many equal-level matches; right tail shows level helps; left tail shows wins despite lower levels.

# 
# Definitions
# bin_start / bin_end: edges of each histogram bin (left-closed, right-open)
# count: number of matches with total_level_diff in [bin_start, bin_end)

# 
# How it’s computed
# numpy.histogram on total_level_diff with bins=40; edges and counts written to CSV.


