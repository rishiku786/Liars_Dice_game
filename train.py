"""
Training by SELF-PLAY  (the Reinforcement Learning part).
=========================================================

Nobody tells the AI the "right" move. Instead:

  1. The network plays a full game AGAINST ITSELF.
  2. We see who won.
  3. REWARD:  winner's moves get +1, loser's moves get -1.
  4. We nudge the network so winning moves become MORE likely and
     losing moves become LESS likely   (this is the REINFORCE algorithm).
  5. Repeat for thousands of games. The AI slowly discovers when to
     raise, when to bluff, and when to call "LIAR!".

Every so often we test the AI against a player that moves at RANDOM and
record the win-rate. Watching that number climb from ~50% toward ~90%
is the "it's learning!" moment for the class.

Run it:
    python train.py                 # ~1-3 minutes on a laptop CPU
Outputs:
    model.pt              the trained network (used by app.py / play.py)
    training_curve.png    the win-rate-vs-random graph for your slides
"""

import argparse
import random
import torch

import game
from model import PolicyNet, save_model


def act(model, my_roll, last_bid, device, greedy=False):
    """
    Pick an action for the player to move. Returns (action, log_prob, entropy).
    Same idea as model.choose_action, but also returns the entropy, which we
    use during training to keep the AI exploring (and willing to bluff).
    """
    state = game.encode(my_roll, last_bid)
    legal = game.legal_actions(last_bid)
    x = torch.tensor(state, dtype=torch.float32, device=device)
    logits = model(x)

    mask = torch.full_like(logits, float("-inf"))
    mask[legal] = 0.0
    logits = logits + mask
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    if greedy:                       # for evaluation: take the best move
        action = torch.argmax(probs)
    else:                            # for training/self-play: sample
        action = dist.sample()
    return action.item(), dist.log_prob(action), dist.entropy()


def play_self_play_game(model, rng, device):
    """
    Play one game, both players controlled by `model`. Returns the list of
    (log_prob, entropy, player) decisions and the winner's index (0 or 1).
    """
    rolls = [game.roll_dice(rng), game.roll_dice(rng)]
    last_bid = None
    player = 0
    history = []                     # (log_prob, entropy, player) per decision

    while True:
        action, log_prob, entropy = act(model, rolls[player], last_bid, device)
        history.append((log_prob, entropy, player))

        if action == game.LIAR:
            # `player` challenges the previous bidder (the other player).
            bidder = 1 - player
            if game.is_bluff(last_bid, rolls[bidder], rolls[1 - bidder]):
                winner = player          # bid was a bluff -> challenger wins
            else:
                winner = bidder          # bid was true   -> bidder wins
            return history, winner

        last_bid = action
        player = 1 - player


def random_opponent_action(last_bid, rng):
    """A baseline opponent that just picks a legal move uniformly at random."""
    return rng.choice(game.legal_actions(last_bid))


def evaluate_vs_random(model, device, n_games=1000, seed=999):
    """Play the trained model against a random player; return its win-rate."""
    rng = random.Random(seed)
    wins = 0
    for g in range(n_games):
        ai_seat = g % 2                        # alternate who goes first
        rolls = [game.roll_dice(rng), game.roll_dice(rng)]
        last_bid = None
        player = 0
        while True:
            if player == ai_seat:
                action, _, _ = act(model, rolls[player], last_bid, device, greedy=True)
            else:
                action = random_opponent_action(last_bid, rng)

            if action == game.LIAR:
                bidder = 1 - player
                if game.is_bluff(last_bid, rolls[bidder], rolls[1 - bidder]):
                    winner = player
                else:
                    winner = bidder
                if winner == ai_seat:
                    wins += 1
                break
            last_bid = action
            player = 1 - player
    return wins / n_games


def train(episodes, batch_size, lr, entropy_coef, eval_every, seed, out_model, out_plot):
    device = "cpu"
    torch.manual_seed(seed)
    rng = random.Random(seed)

    model = PolicyNet()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    history_x, history_y = [], []            # for the win-rate plot

    # Baseline before any training, so the slide shows the starting point.
    wr = evaluate_vs_random(model, device)
    history_x.append(0)
    history_y.append(wr)
    print(f"episode      0 | win-rate vs random: {wr:5.1%}  (untrained)")

    batch_loss = []
    for ep in range(1, episodes + 1):
        history, winner = play_self_play_game(model, rng, device)

        # REINFORCE: each decision's reward is +1 if that player won, else -1.
        loss = 0.0
        for log_prob, entropy, player in history:
            reward = 1.0 if player == winner else -1.0
            # maximise reward*log_prob  ==  minimise  -reward*log_prob
            # minus entropy term keeps the strategy from collapsing too early.
            loss = loss - reward * log_prob - entropy_coef * entropy
        batch_loss.append(loss)

        # Update the network once per batch of games (more stable).
        if ep % batch_size == 0:
            optim.zero_grad()
            (torch.stack(batch_loss).mean()).backward()
            optim.step()
            batch_loss = []

        if ep % eval_every == 0:
            wr = evaluate_vs_random(model, device)
            history_x.append(ep)
            history_y.append(wr)
            print(f"episode {ep:6d} | win-rate vs random: {wr:5.1%}")

    save_model(model, out_model)
    print(f"\nSaved trained model -> {out_model}")

    # Draw the win-rate curve for the slides (skip gracefully if no matplotlib).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(history_x, [y * 100 for y in history_y], marker="o")
        plt.axhline(50, color="gray", linestyle="--", label="random (50%)")
        plt.xlabel("self-play games trained")
        plt.ylabel("win-rate vs random player (%)")
        plt.title("Liar's Dice AI learning by self-play")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_plot, dpi=120)
        print(f"Saved training curve -> {out_plot}")
    except Exception as e:
        print(f"(Could not draw plot: {e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Liar's Dice AI by self-play.")
    p.add_argument("--episodes", type=int, default=60000, help="self-play games")
    p.add_argument("--batch-size", type=int, default=32, help="games per update")
    p.add_argument("--lr", type=float, default=0.002, help="learning rate")
    p.add_argument("--entropy", type=float, default=0.02, help="exploration bonus")
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-model", type=str, default="model.pt")
    p.add_argument("--out-plot", type=str, default="training_curve.png")
    args = p.parse_args()

    train(args.episodes, args.batch_size, args.lr, args.entropy,
          args.eval_every, args.seed, args.out_model, args.out_plot)
