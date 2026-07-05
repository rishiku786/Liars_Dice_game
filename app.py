"""
The browser game server.
========================

Run this, then open http://localhost:8000 in your browser to play
Liar's Dice against the trained AI.

    python app.py

How it fits together:
    Browser (play.html)  --HTTP-->  this server  -->  model.pt (PyTorch AI)

We use Python's built-in http.server so there is NOTHING extra to install
beyond torch. The whole game state lives here on the server; the browser
just shows dice and buttons and sends the human's moves.
"""

import json
import os
import random
import http.server

import torch

import game
from model import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pt")

if not os.path.exists(MODEL_PATH):
    raise SystemExit("model.pt not found — run 'python train.py' first.")

MODEL = load_model(MODEL_PATH)
RNG = random.Random()

# The single shared game (one browser at a time — fine for a classroom demo).
STATE = {}


def ai_decide(ai_roll, last_bid):
    """
    The AI chooses its move AND we capture its 'thinking' (the probabilities
    it assigned) so we can show it on screen — a nice teaching moment.
    """
    state = game.encode(ai_roll, last_bid)
    legal = game.legal_actions(last_bid)
    x = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        logits = MODEL(x)
    mask = torch.full_like(logits, float("-inf"))
    mask[legal] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)

    action = int(torch.argmax(probs))            # greedy: its best move
    # Top few options it considered, for the "AI thinking" panel.
    ranked = sorted(legal, key=lambda a: -probs[a].item())
    thinking = [
        {"action": game.action_name(a), "prob": round(probs[a].item() * 100, 1)}
        for a in ranked[:4]
    ]
    return action, thinking


def new_game(human_first=None):
    if human_first is None:
        human_first = RNG.random() < 0.5
    human_seat = 0 if human_first else 1
    STATE.clear()
    STATE.update(
        human_seat=human_seat,
        ai_seat=1 - human_seat,
        rolls=[game.roll_dice(RNG), game.roll_dice(RNG)],
        last_bid=None,
        turn=0,                      # player 0 always acts first
        over=False,
        log=[],
    )
    ai_move_if_needed()
    return public_state()


def resolve(challenger):
    """`challenger` (a seat) called LIAR on the previous bid. Finish the game."""
    bidder = 1 - challenger
    bluff = game.is_bluff(STATE["last_bid"], STATE["rolls"][bidder],
                          STATE["rolls"][1 - bidder])
    winner = challenger if bluff else bidder
    STATE["over"] = True
    STATE["winner"] = winner
    STATE["bluff"] = bluff
    who = "You" if challenger == STATE["human_seat"] else "The AI"
    verdict = "a BLUFF" if bluff else "TRUE"
    STATE["log"].append(f'{who} called LIAR! The bid was {verdict}.')


def ai_move_if_needed():
    """If it's the AI's turn (and game not over), let it act — possibly twice
    in a row is impossible; it acts once then control returns to the human."""
    if STATE["over"]:
        return
    if STATE["turn"] != STATE["ai_seat"]:
        return
    action, thinking = ai_decide(STATE["rolls"][STATE["ai_seat"]], STATE["last_bid"])
    STATE["ai_thinking"] = thinking
    if action == game.LIAR:
        STATE["log"].append("The AI says: LIAR!")
        resolve(STATE["ai_seat"])
    else:
        STATE["log"].append(f"The AI bids: {game.action_name(action)}")
        STATE["last_bid"] = action
        STATE["turn"] = STATE["human_seat"]


def public_state():
    """What we send to the browser. Never reveals the AI's dice until game over."""
    human_seat = STATE["human_seat"]
    legal = [] if (STATE["over"] or STATE["turn"] != human_seat) \
        else game.legal_actions(STATE["last_bid"])
    return {
        "your_dice": list(STATE["rolls"][human_seat]),
        "your_turn": (not STATE["over"]) and STATE["turn"] == human_seat,
        "last_bid": None if STATE["last_bid"] is None
        else {"id": STATE["last_bid"], "name": game.action_name(STATE["last_bid"])},
        "legal": [{"id": a, "name": game.action_name(a)} for a in legal],
        "liar_id": game.LIAR,
        "over": STATE["over"],
        "winner": ("you" if STATE.get("winner") == human_seat else "ai")
        if STATE["over"] else None,
        "ai_dice": list(STATE["rolls"][STATE["ai_seat"]]) if STATE["over"] else None,
        "ai_thinking": STATE.get("ai_thinking", []),
        "log": STATE["log"],
        "sides": game.SIDES,
        "dice_per_player": game.DICE_PER_PLAYER,
    }


def human_move(action):
    if STATE["over"] or STATE["turn"] != STATE["human_seat"]:
        return public_state()
    if action == game.LIAR:
        STATE["log"].append("You call LIAR!")
        resolve(STATE["human_seat"])
    else:
        STATE["log"].append(f"You bid: {game.action_name(action)}")
        STATE["last_bid"] = action
        STATE["turn"] = STATE["ai_seat"]
        ai_move_if_needed()
    return public_state()


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html", "/play.html"):
            with open(os.path.join(HERE, "play.html"), "rb") as f:
                self._send(200, f.read(), "text/html")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or "{}")
        if self.path == "/api/new":
            self._send(200, json.dumps(new_game(body.get("human_first"))))
        elif self.path == "/api/move":
            self._send(200, json.dumps(human_move(int(body["action"]))))
        else:
            self._send(404, json.dumps({"error": "unknown endpoint"}))

    def log_message(self, *args):
        pass                          # keep the console quiet


if __name__ == "__main__":
    import sys
    # Port: default 8000, override with `python app.py 8123` or PORT env var.
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8000))
    print(f"\n  Liar's Dice is running!  Open  ->  http://localhost:{port}\n")
    http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
