"""
Liar's Dice — the game environment (1 vs 1).
================================================

This file knows ONLY the rules of the game. It has no AI in it.
The AI (a neural network) lives in model.py and learns by playing
this environment against itself in train.py.

------------------------------------------------------------------
HOW THE GAME WORKS (simplified "Dudo")
------------------------------------------------------------------
- There are two players. Each secretly rolls DICE_PER_PLAYER dice,
  each die showing 1..SIDES.
- Players take turns making a "bid". A bid is a claim about ALL the
  dice on the table (yours + the opponent's, which you can't see):
      "(count, face)"  means  "there are at least <count> dice
                               showing <face> in total".
- Each new bid must be strictly HIGHER than the previous one:
  a bigger count, or the same count with a bigger face.
- Instead of bidding, a player may shout "LIAR!" to challenge the
  last bid. Then all dice are revealed:
      * if the real number of <face> dice is >= <count> -> the bid
        was TRUE, so the challenger loses.
      * otherwise the bid was a bluff, so the bidder loses.
- The very first player must bid (you can't challenge nothing).

That's it. The whole strategy is: "Given my dice and the bidding so
far, should I raise the bid (maybe bluffing) or call the opponent's
bluff?" — a decision under uncertainty. That is what the AI learns.
"""

import itertools
from collections import Counter

# ---- Game size. Small on purpose so training finishes in minutes. ----
DICE_PER_PLAYER = 2      # each player rolls this many dice
SIDES = 6               # a normal 6-sided die
TOTAL_DICE = DICE_PER_PLAYER * 2

# A "bid" is (count, face). count runs 1..TOTAL_DICE, face runs 1..SIDES.
# We give every possible bid a single integer id so the network can output
# one number per action. Ordering is by count first, then face, so that
# a bigger id always means a strictly higher bid.
#   id = (count - 1) * SIDES + (face - 1)
NUM_BIDS = TOTAL_DICE * SIDES          # every possible (count, face)
LIAR = NUM_BIDS                        # the "call liar" action gets the last id
NUM_ACTIONS = NUM_BIDS + 1             # all bids + the LIAR action


def bid_to_id(count, face):
    return (count - 1) * SIDES + (face - 1)


def id_to_bid(action):
    """Turn an action id back into (count, face). Not valid for LIAR."""
    count, face = divmod(action, SIDES)
    return count + 1, face + 1


def action_name(action):
    """Human-readable name of an action, e.g. '3 fours' or 'LIAR'."""
    if action == LIAR:
        return "LIAR!"
    count, face = id_to_bid(action)
    face_word = {1: "ones", 2: "twos", 3: "threes",
                 4: "fours", 5: "fives", 6: "sixes"}[face]
    return f"{count} {face_word}"


def roll_dice(rng):
    """Roll DICE_PER_PLAYER dice, return them sorted as a tuple."""
    return tuple(sorted(rng.randint(1, SIDES) for _ in range(DICE_PER_PLAYER)))


def legal_actions(last_bid):
    """
    Which actions may the current player take?
    - last_bid is None  -> the game just started: must make any bid, cannot
                           call LIAR yet.
    - last_bid is an id -> may make any strictly higher bid, or call LIAR.
    Returns a list of action ids.
    """
    if last_bid is None:
        return list(range(NUM_BIDS))            # any bid, no LIAR
    return list(range(last_bid + 1, NUM_BIDS)) + [LIAR]


def is_bluff(last_bid, my_roll, opp_roll):
    """
    Resolve a challenge. Returns True if the bid was a BLUFF (i.e. the
    challenger who called LIAR wins), False if the bid was TRUE.
    """
    count, face = id_to_bid(last_bid)
    actual = Counter(my_roll + opp_roll)[face]   # how many <face> really exist
    return actual < count                        # bluff if reality falls short


# ------------------------------------------------------------------
# State encoding: turning the situation into numbers for the network.
# ------------------------------------------------------------------
# The network sees a fixed-length vector made of:
#   [ my dice histogram (SIDES numbers) ]         <- private info
#   [ one-hot of the current highest bid (NUM_BIDS) ]  <- public info
#   [ 1 number: 1 if no bid has been made yet, else 0 ]
INPUT_SIZE = SIDES + NUM_BIDS + 1


def encode(my_roll, last_bid):
    """Build the network input vector for the player about to act."""
    vec = [0.0] * INPUT_SIZE
    # my dice, as a histogram: how many of each face I hold
    for die in my_roll:
        vec[die - 1] += 1.0
    # the current bid on the table
    if last_bid is None:
        vec[SIDES + NUM_BIDS] = 1.0              # "no bid yet" flag
    else:
        vec[SIDES + last_bid] = 1.0              # one-hot the current bid
    return vec


if __name__ == "__main__":
    # Tiny self-check so you can see the rules working.
    import random
    rng = random.Random(0)
    r1, r2 = roll_dice(rng), roll_dice(rng)
    print("Player 1 rolled:", r1)
    print("Player 2 rolled:", r2)
    print("Total dice on table:", TOTAL_DICE, " Possible bids:", NUM_BIDS)
    print("First player's legal actions:", [action_name(a) for a in legal_actions(None)][:5], "...")
    b = bid_to_id(2, 6)                            # "2 sixes"
    print(f"After a bid of '{action_name(b)}', legal replies:",
          [action_name(a) for a in legal_actions(b)])
    print(f"Is '{action_name(b)}' a bluff?  ->", is_bluff(b, r1, r2))
