# Comments
# - deck_overlap measures how many cards both decks share (0–8).
# - Overlap=8 are mirror matches; high overlap indicates meta convergence, low overlap indicates diversity.

# 
# Definitions
# deck_overlap: |set(winner.cards.list) ∩ set(loser.cards.list)|
# count: number of matches with that exact overlap

# 
# How it’s computed
# For each match, take the set intersection size of winner and loser decks, then tally with value_counts().


