# Comments
# - “Upset” = winner had fewer starting trophies. By definition, negative trophy_gap bins show high upset rates.
# - For a more informative view, bin by |trophy_gap| and report upset rate vs gap magnitude (how often underdogs win as the gap grows).

# 
# Definitions
# trophy_gap: winner.startingTrophies − loser.startingTrophies (then binned)
# upset: mean(winner had fewer trophies) within each bin

# 
# How it’s computed
# Bin trophy_gap into ranges, then groupby bin and take the mean of (trophy_gap < 0).


