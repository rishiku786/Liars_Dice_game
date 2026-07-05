"""
The AI's brain: a small neural network (a "policy network").
============================================================

This is the DEEP part of Deep Reinforcement Learning.

- INPUT : the game situation as numbers (my dice + the current bid),
          produced by game.encode().
- OUTPUT: one number ("logit") for every possible action. After we hide
          the illegal actions and apply softmax, these become the
          PROBABILITIES of taking each action — i.e. the AI's STRATEGY.

The network starts out random (it plays nonsense). Training in train.py
nudges its weights so that actions which led to WINS become more likely
and actions that led to LOSSES become less likely. That is how it learns.
"""

import torch
import torch.nn as nn

from game import INPUT_SIZE, NUM_ACTIONS


class PolicyNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # A plain feed-forward net: input -> hidden -> hidden -> one score per action.
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )

    def forward(self, x):
        # Returns raw scores (logits), one per action. Masking + softmax
        # happen in choose_action() so the network stays simple.
        return self.net(x)


def choose_action(model, state_vec, legal, device="cpu"):
    """
    Ask the network what to do.

    state_vec : list[float]  from game.encode(...)
    legal     : list[int]    legal action ids from game.legal_actions(...)

    Returns (action_id, log_prob, probs_dict)
      action_id : the sampled action
      log_prob  : log-probability of that choice (needed for training)
      probs_dict: {action_id: probability} over legal actions (for display)
    """
    x = torch.tensor(state_vec, dtype=torch.float32, device=device)
    logits = model(x)

    # Hide illegal actions by pushing their score to -infinity, so softmax
    # gives them probability 0.
    mask = torch.full_like(logits, float("-inf"))
    mask[legal] = 0.0
    logits = logits + mask

    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()                       # sample -> lets the AI bluff & vary
    log_prob = dist.log_prob(action)

    probs_dict = {a: probs[a].item() for a in legal}
    return action.item(), log_prob, probs_dict


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path, device="cpu"):
    model = PolicyNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
