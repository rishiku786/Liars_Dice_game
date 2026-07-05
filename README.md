# 🎲 Liar's Dice by Self-Play — a Reinforcement Learning case study

An AI that teaches **itself** to play (and bluff at) Liar's Dice, using
**Deep Reinforcement Learning** in PyTorch. It starts knowing nothing and
learns purely by playing **60,000 games against itself** — in about
**30 seconds** on a normal laptop CPU (no GPU needed).

Built as a teaching example: small, self-contained, and fully runnable.

---

## What is Liar's Dice?

A bluffing game of hidden information (a cousin of Poker).

1. Two players each secretly roll **2 dice**.
2. They take turns making a **bid** — a claim about **all 4 dice on the table**,
   e.g. *"three fours"* means *"there are at least three 4s in total"*.
3. Each new bid must be **strictly higher** than the previous one.
4. Instead of bidding, a player can shout **"LIAR!"** to challenge the last bid.
   All dice are revealed:
   - if the real count meets the bid → the bid was **true**, the challenger loses;
   - otherwise it was a **bluff**, the bidder loses.

Because you can't see the opponent's dice, every move is a **decision under
uncertainty** — the perfect playground for reinforcement learning.

---

## ML vs DL vs RL (the concept)

- **Machine Learning (ML)** — learn patterns from data/examples.
- **Deep Learning (DL)** — ML using **neural networks**.
- **Reinforcement Learning (RL)** — an **agent** takes **actions**, gets a
  **reward**, and learns by trial-and-error what leads to winning.

This project is **Deep RL = DL + RL**: a neural network is the brain, trained by
the reward of winning games.

### RL pieces, mapped onto this game

| RL concept  | In Liar's Dice                                   |
|-------------|--------------------------------------------------|
| Agent       | the AI player (a small neural network)           |
| Environment | the dice game + the opponent                     |
| State       | my 2 dice + the bids made so far                 |
| Action      | raise the bid, or call "LIAR!"                   |
| Reward      | +1 if the agent wins the round, −1 if it loses   |
| Policy      | the learned strategy: state → which action       |
| Self-play   | the agent plays itself thousands of times        |

The learning rule (the **REINFORCE** algorithm): after each game, make the
**winner's** moves more likely and the **loser's** moves less likely. Repeat.

---

## The files

| File                 | What it is                                                        |
|----------------------|-------------------------------------------------------------------|
| `game.py`            | The **rules only** — dice, bids, legal moves, who wins a challenge |
| `model.py`           | The AI's brain — a small **PyTorch** neural network (the *policy*) |
| `train.py`           | **Self-play training** loop; saves `model.pt` and the win-rate plot |
| `app.py` + `play.html` | Play the AI in your **browser** (the main demo)                 |
| `play.py`            | Play the AI in the **terminal** (no browser needed)               |
| `slides.html`        | A 5-slide teaching deck (open in a browser, use ← → arrow keys)    |
| `model.pt`           | The trained network (created by `train.py`)                       |
| `training_curve.png` | The "it's learning!" graph (created by `train.py`)                |

---

## How to run

**1. Install the two dependencies** (one-time):

```bash
pip install torch matplotlib
```

**2. Train the AI** (~30 seconds on CPU). Watch the win-rate climb:

```bash
python train.py
```

You'll see something like:

```
episode      0 | win-rate vs random: 24.3%  (untrained)
episode   5000 | win-rate vs random: 91.5%
...
episode  60000 | win-rate vs random: 92.3%
```

This creates `model.pt` (the trained AI) and `training_curve.png` (the graph).

> A trained `model.pt` is already included, so you can skip straight to playing
> if you like — but re-training live is the best part of the demo.

**3. Play against it — in the browser** (recommended):

```bash
python app.py
```

Then open **http://localhost:8000** and play.
(Port 8000 busy? Use another: `python app.py 8123`.)

**...or in the terminal:**

```bash
python play.py
```

**4. Show the slides:** open `slides.html` in any browser and use the ← → arrow keys.

---

## Notes for teaching

- **Everything is small on purpose.** 2 dice per player, 6 sides — so training
  finishes in seconds and the whole state fits in your head.
- The browser game shows a live **"what the AI is thinking"** panel (its action
  probabilities) — a nice moment to explain that the network's output *is* the
  strategy.
- Untrained, the network starts **below 50%** (it stubbornly repeats one bad
  move). That contrast makes the learning curve dramatic.
- Want to make it harder/easier? Change `DICE_PER_PLAYER` or `SIDES` at the top
  of `game.py`, then re-run `python train.py`.

---

*Inspired by Thomas Ahle's original [liars-dice](https://github.com/thomasahle/liars-dice)
self-play project. This is a fresh, minimal re-build for teaching — our own game
engine, our own PyTorch model, trained from scratch.*
