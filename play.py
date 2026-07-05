"""
Play Liar's Dice against the trained AI — in the TERMINAL.
==========================================================

This is the no-browser version (handy for showing the code, or if you
just want a quick game in the console).

    python play.py

You'll be shown your dice, and each turn you either raise the bid or
type 'liar' to challenge the AI.
"""

import random
import torch

import game
from model import load_model, choose_action


def show(roll):
    return " ".join(str(d) for d in roll)


def main():
    model = load_model("model.pt")
    rng = random.Random()

    print("\n=== Liar's Dice — you vs. the self-taught AI ===")
    print(f"Each player has {game.DICE_PER_PLAYER} dice ({game.SIDES} sides). "
          f"A bid is a claim about ALL {game.TOTAL_DICE} dice.\n")

    while True:
        human_first = rng.random() < 0.5
        human_seat = 0 if human_first else 1
        ai_seat = 1 - human_seat
        rolls = [game.roll_dice(rng), game.roll_dice(rng)]
        last_bid = None
        turn = 0

        print("-" * 48)
        print(f"Your dice: [ {show(rolls[human_seat])} ]   "
              f"(you go {'first' if human_first else 'second'})")

        winner = None
        while winner is None:
            if turn == human_seat:
                legal = game.legal_actions(last_bid)
                # Show the human what they can do.
                if last_bid is not None:
                    print(f"\nAI's current bid: {game.action_name(last_bid)}")
                while True:
                    raw = input("Your move — a bid like '2 fives', or 'liar': ").strip().lower()
                    if raw in ("liar", "l"):
                        if last_bid is None:
                            print("  You can't call liar on the opening move.")
                            continue
                        action = game.LIAR
                        break
                    parts = raw.split()
                    words = {"ones": 1, "twos": 2, "threes": 3, "fours": 4,
                             "fives": 5, "sixes": 6}
                    if len(parts) == 2 and parts[0].isdigit() and parts[1] in words:
                        count, face = int(parts[0]), words[parts[1]]
                        if 1 <= count <= game.TOTAL_DICE:
                            action = game.bid_to_id(count, face)
                            if action in legal:
                                break
                        print("  That bid isn't higher than the current one — try again.")
                    else:
                        print("  Format: '<number> <face>' e.g. '2 fours', or 'liar'.")
            else:
                # AI's turn — greedy best move.
                action, _, probs = choose_action(
                    model, game.encode(rolls[ai_seat], last_bid),
                    game.legal_actions(last_bid))
                action = max(probs, key=probs.get)     # greedy
                print(f"\nThe AI {'bids ' + game.action_name(action) if action != game.LIAR else 'calls LIAR!'}")

            if action == game.LIAR:
                challenger = turn
                bidder = 1 - turn
                bluff = game.is_bluff(last_bid, rolls[bidder], rolls[1 - bidder])
                winner = challenger if bluff else bidder
                print("\n>>> Reveal!")
                print(f"    You had:    [ {show(rolls[human_seat])} ]")
                print(f"    AI had:     [ {show(rolls[ai_seat])} ]")
                print(f"    The bid '{game.action_name(last_bid)}' was "
                      f"{'a BLUFF' if bluff else 'TRUE'}.")
                break

            last_bid = action
            turn = 1 - turn

        print("\n*** YOU WIN! ***" if winner == human_seat else "\n*** THE AI WINS! ***")
        if input("\nPlay again? [y/n] ").strip().lower() not in ("y", "yes", ""):
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
