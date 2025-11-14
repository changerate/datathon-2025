# Comments
# - Histogram of elixir_diff = winner.elixir.average − loser.elixir.average.
# - Distribution is centered near 0: winners and losers run similar average elixir, reinforcing that deck weight alone doesn’t decide matches.

# 
# Definitions
# bin_start / bin_end: edges of each histogram bin (left-closed, right-open)
# count: number of matches with elixir_diff in [bin_start, bin_end)

# 
# How it’s computed
# numpy.histogram on elixir_diff with bins=40; bin edges and counts are written to the CSV.





